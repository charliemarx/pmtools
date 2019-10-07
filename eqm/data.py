from copy import deepcopy
from pathlib import Path
from imblearn.over_sampling import RandomOverSampler
import warnings
import itertools
import numpy as np
import pandas as pd
from eqm.cross_validation import validate_cvindices, validate_folds, to_fold_id, validate_fold_id, is_inner_fold_id

from eqm.debug import ipsh

# Constants
INTERCEPT_IDX = 0
INTERCEPT_NAME = '(Intercept)'

OUTCOME_NAME = 'Y'
OUTCOME_VALUES = {-1, 1}
POSITIVE_LABEL = '+1'
NEGATIVE_LABEL = '-1'
FORMAT_NAME_DEFAULT = 'standard'

#
RAW_DATA_OUTCOME_COL_IDX = 0
RAW_DATA_FILE_VALID_EXT = {'csv', 'data'}
RAW_HELPER_FILE_VALID_EXT = {'csv', 'helper'}
RAW_WEIGHTS_FILE_VALID_EXT = {'csv', 'weights'}
PROCESSED_DATA_FILE_RDATA = {'rdata'}
PROCESSED_DATA_FILE_PICKLE = {'p', 'pk', 'pickle'}
PROCESSED_DATA_FILE_VALID_EXT = PROCESSED_DATA_FILE_RDATA.union(PROCESSED_DATA_FILE_PICKLE)


def load_data_from_csv(dataset_csv_file, sample_weights_csv_file = None, fold_csv_file = None, fold_num = 0):
    """

    Parameters
    ----------
    dataset_csv_file                csv file containing the training data
                                    see /datasets/adult_data.csv for an example
                                    training data stored as a table with N + 1 rows and d+1 columns
                                    column 1 is the outcome variable entries must be (-1,1) or (0,1)
                                    column 2 to d+1 are the d input variables
                                    row 1 contains unique names for the outcome variable, and the input vairable

    sample_weights_csv_file         csv file containing sample weights for the training data
                                    weights stored as a table with N rows and 1 column
                                    all sample weights must be non-negative

    fold_csv_file                   csv file containing indices of folds for K-fold cross validation
                                    fold indices stored as a table with N rows and 1 column
                                    folds must be integers between 1 to K
                                    if fold_csv_file is None, then we do not use folds

    fold_num                        int between 0 to K, where K is set by the fold_csv_file
                                    let fold_idx be the N x 1 index vector listed in fold_csv_file
                                    samples where fold_idx == fold_num will be used to test
                                    samples where fold_idx != fold_num will be used to train the model
                                    fold_num = 0 means use "all" of the training data (since all values of fold_idx \in [1,K])
                                    if fold_csv_file is None, then fold_num is set to 0


    Returns
    -------
    dictionary containing training data for a binary classification problem with the fields:

     - 'X' N x P matrix of features (numpy.ndarray) with a column of 1s for the '(Intercept)'
     - 'Y' N x 1 vector of labels (+1/-1) (numpy.ndarray)
     - 'variable_names' list of strings containing the names of each feature (list)
     - 'Y_name' string containing the name of the output (optional)
     - 'sample_weights' N x 1 vector of sample weights, must all be positive

    """

    dataset_csv_file = Path(dataset_csv_file)
    if dataset_csv_file.exists():
        df = pd.read_csv(dataset_csv_file, sep=',')
    else:
        raise IOError('could not find dataset_csv_file: %s' % dataset_csv_file)

    raw_data = df.values
    data_headers = list(df.columns.values)
    N = raw_data.shape[0]

    # setup Y vector and Y_name
    Y_col_idx = [0]
    Y = raw_data[:, Y_col_idx]
    Y_name = data_headers[Y_col_idx[0]]
    Y[Y == 0] = -1
    Y = Y.flatten()

    # setup X and X_names
    X_col_idx = [j for j in range(raw_data.shape[1]) if j not in Y_col_idx]
    X = raw_data[:, X_col_idx]
    variable_names = [data_headers[j] for j in X_col_idx]

    # insert a column of ones to X for the intercept
    if sample_weights_csv_file is None:
        sample_weights = np.ones(N)
    else:
        sample_weights_csv_file = Path(sample_weights_csv_file)
        if sample_weights_csv_file.exists():
            sample_weights = pd.read_csv(sample_weights_csv_file, sep=',', header = None)
            sample_weights = sample_weights.values
        else:
            raise IOError('could not find sample_weights_csv_file: %s' % sample_weights_csv_file)


    data = {
        'X': X,
        'Y': Y,
        'variable_names': variable_names,
        'outcome_name': Y_name,
        'sample_weights': sample_weights,
        }

    #load folds
    if fold_csv_file is not None:
        fold_csv_file = Path(fold_csv_file)
        if not fold_csv_file.exists():
            raise IOError('could not find fold_csv_file: %s' % fold_csv_file)
        else:
            fold_idx = pd.read_csv(fold_csv_file, sep=',', header=None)
            fold_idx = fold_idx.values.flatten()
            K = max(fold_idx)
            all_fold_nums = np.sort(np.unique(fold_idx))
            assert len(fold_idx) == N, "dimension mismatch: read %r fold indices (expected N = %r)" % (len(fold_idx), N)
            assert np.all(all_fold_nums == np.arange(1, K+1)), "folds should contain indices between 1 to %r" % K
            assert fold_num in np.arange(0, K+1), "fold_num should either be 0 or an integer between 1 to %r" % K
            if fold_num >= 1:
                test_idx = fold_num == fold_idx
                train_idx = fold_num != fold_idx
                data['X'] = data['X'][train_idx,]
                data['Y'] = data['Y'][train_idx]
                data['sample_weights'] = data['sample_weights'][train_idx]

    data = add_intercept(data)
    assert check_data(data)
    return data


def check_data(data):
    """
    makes sure that 'data' contains training data that is suitable for binary classification problems
    throws AssertionError if

    'data' is a dictionary that must contain:

     - 'X' N x P matrix of features (numpy.ndarray) with a column of 1s for the '(Intercept)'
     - 'Y' N x 1 vector of labels (+1/-1) (numpy.ndarray)
     - 'variable_names' list of strings containing the names of each feature (list)

     data can also contain:

     - 'outcome_name' string containing the name of the output (optional)
     - 'sample_weights' N x 1 vector of sample weights, must all be positive

    Returns
    -------
    True if data passes checks

    """
    # type checks
    assert isinstance(data, dict), "data should be a dict"

    assert 'X' in data, "data should contain X matrix"
    assert isinstance(data['X'], np.ndarray), "type(X) should be numpy.ndarray"

    assert 'Y' in data, "data should contain Y matrix"
    assert isinstance(data['Y'], np.ndarray), "type(Y) should be numpy.ndarray"

    assert 'variable_names' in data, "data should contain variable_names"
    assert isinstance(data['variable_names'], list), "variable_names should be a list"

    X = data['X']
    Y = data['Y']
    variable_names = data['variable_names']

    if 'outcome_name' in data:
        assert isinstance(data['outcome_name'], str), "outcome_name should be a str"

    # sizes and uniqueness
    N, P = X.shape
    assert N > 0, 'X matrix must have at least 1 row'
    assert P > 0, 'X matrix must have at least 1 column'
    assert len(Y) == N, 'dimension mismatch. Y must contain as many entries as X. Need len(Y) = N.'
    assert len(list(set(data['variable_names']))) == len(data['variable_names']), 'variable_names is not unique'
    assert len(data['variable_names']) == P, 'len(variable_names) should be same as # of cols in X'

    # feature matrix
    assert np.all(~np.isnan(X)), 'X has nan entries'
    assert np.all(~np.isinf(X)), 'X has inf entries'

    # offset in feature matrix
    if INTERCEPT_NAME in variable_names:
        assert all(X[:, variable_names.index('(Intercept)')] == 1.0), "%s' column should only be composed of 1s" % INTERCEPT_NAME
    else:
        warnings.warn("there is no column named '%s' in variable_names" % INTERCEPT_NAME)

    # labels values
    assert np.isin(Y, list(OUTCOME_VALUES)).all(), 'need Y[i] = [-1,1] for all i.'
    if all(Y == 1):
        warnings.warn('Y does not contain any positive examples. Need Y[i] = +1 for at least 1 i.')
    if all(Y == -1):
        warnings.warn('Y does not contain any negative examples. Need Y[i] = -1 for at least 1 i.')

    if 'sample_weights' in data:
        sample_weights = data['sample_weights']
        assert isinstance(sample_weights, np.ndarray)
        assert all(sample_weights > 0.0), 'sample_weights[i] > 0 for all i '
        assert len(sample_weights) == N, 'sample_weights should contain N elements'

        # by default, we set sample_weights as an N x 1 array of ones. if not, then sample weights is non-trivial
        if any(sample_weights != 1.0) and len(np.unique(sample_weights)) < 2:
            warnings.warn('note: sample_weights only has 1 unique value')

    return True


def check_data_required_fields(X, Y, variable_names, ready_for_training = False, **args):
    """
        makes sure that 'data' contains training data that is suitable for binary classification problems
        throws AssertionError if

        'data' is a dictionary that must contain:

         - 'X' N x P matrix of features (numpy.ndarray) with a column of 1s for the '(Intercept)'
         - 'Y' N x 1 vector of labels (+1/-1) (numpy.ndarray)
         - 'variable_names' list of strings containing the names of each feature (list)

         data can also contain:

         - 'outcome_name' string containing the name of the output (optional)
         - 'sample_weights' N x 1 vector of sample weights, must all be positive

        Returns
        -------
        True if data passes checks

        """
    # type checks

    assert isinstance(X, np.ndarray), \
        "type(X) should be numpy.ndarray"

    assert isinstance(Y, np.ndarray), \
        "type(Y) should be numpy.ndarray"

    assert isinstance(variable_names, list),\
        "variable_names should be a list"

    assert len(variable_names) == len(set(variable_names)), \
        'variable_names is not unique'

    # if it's ready for training then it should be numeric
    if ready_for_training:
        assert np.can_cast(X.dtype, np.float, casting = 'safe')
        assert np.can_cast(Y.dtype, np.float, casting = 'safe')

    # labels values
    assert np.all(np.isin(Y, (-1, 1))), \
        'Y must be binary for all i'

    if all(Y == 1):
        warnings.warn('Y does not contain any positive examples. need Y[i] = +1 for at least 1 i.')

    if all(Y == -1):
        warnings.warn('Y does not contain any negative examples. need Y[i] = -1 for at least 1 i.')

    # sizes and uniqueness
    n_variables = len(variable_names)
    n, d = X.shape

    assert n > 0, \
        'X matrix must have at least 1 row'

    assert d > 0, \
        'X matrix must have at least 1 column'

    assert len(Y) == n, \
        'dimension mismatch. Y must contain as many entries as X. Need len(Y) = N.'

    assert n_variables == d, \
        'len(variable_names) should be same as # of cols in X'

    return True


def set_defaults_for_data(data):

    data.setdefault('format', FORMAT_NAME_DEFAULT)
    data.setdefault('outcome_name', OUTCOME_NAME)
    data.setdefault('outcome_label_positive', POSITIVE_LABEL)
    data.setdefault('outcome_label_negative', NEGATIVE_LABEL)

    for xf, yf, sw in [('X', 'Y', 'sample_weights'),
                       ('X_test', 'Y_test', 'sample_weights_test'),
                       ('X_validation', 'Y_validation', 'sample_weights_validation')]:

        if xf and yf in data:

            n_points = data[xf]
            data[yf] = data[yf].flatten()
            if sw in data:
                if data[sw].ndim > 1 and data[sw].shape == (n_points, 1):
                    data[sw] = data[sw].flatten()
            else:
                data[sw] = np.ones(n_points, dtype = np.float)

    return data


#### views  ####

def has_test_set(data):
    return 'X_test' in data and 'Y_test' in data


def has_validation_set(data):
    return 'X_validation' in data and 'Y_validation' in data


def has_sample_weights(data):
    return 'sample_weights' in data and np.any(data['sample_weights'] != 1.0)


#### add / remove variables ####

def rename_variable(data, name, new_name):
    assert name in data['variable_names']
    idx = data['variable_names'].index(name)

    data['variable_names'][idx] = new_name
    data['variable_types'][new_name] = data['variable_types'].pop(name)

    if name in data['variable_orderings']:
        data['variable_orderings'][new_name] = data['variable_orderings'].pop(name)

    if 'partitions' in data and name in data['partitions']:
        part_idx = data['partitions'].index(name)
        data['partitions'][part_idx] = new_name

    return data


def add_variable(data, values, name, idx = None, test_values = None, validation_values = None):


    assert isinstance(name, str), \
        'name must be str'

    assert len(name) > 0, \
        'name must have at least 1 character'

    assert name not in data['variable_names'], \
        'data already contains a variable with name %s already exists' % name


    n_variables = len(data['variable_names'])
    idx = n_variables if idx is None else int(idx)
    assert idx in range(n_variables + 1)


    # add values first
    for vals, field in [(values, 'X'), (test_values, 'X_test'), (validation_values, 'X_validation')]:

        if field in data:

            assert vals is not None
            vals = np.array(vals).flatten()
            assert len(vals) in (1, data[field].shape[0]), \
                'invalid shape for %s values (must be scalar or array with same length)' % field

            data[field] = np.insert(arr = data[field], values = vals, obj = idx, axis = 1)

    data['variable_names'].insert(idx, name)

    assert check_data(data)
    return data


def remove_variable(data, name):

    #check that name exists (will throw ValueError otherwise)
    data = deepcopy(data)
    idx = data['variable_names'].index(name)

    # remove entries from feature matrix
    data['X'] = np.delete(data['X'], idx, axis = 1)

    #remove fields
    data['variable_names'].remove(name)

    if 'variable_types' in data:
        data['variable_types'].pop(name)

    if 'variable_orderings' in data and name in data['variable_orderings']:
        data['variable_orderings'].pop(name)

    if 'partitions' in data and name in data['partitions']:
        data['partitions'].remove(name)

    if has_test_set(data):
        data['X_test'] = np.delete(data['X_test'], idx, axis = 1)

    if has_validation_set(data):
        data['X_validation'] = np.delete(data['X_validation'], idx, axis = 1)

    return data


#### add /remove intercept ####


def get_intercept_index(data):
    try:
        return data['variable_names'].index(INTERCEPT_NAME)
    except ValueError:
        return -1


def has_intercept(data):
    idx = get_intercept_index(data)
    if idx == -1:
        return False
    else:
        assert check_intercept(data)
        return True


def check_intercept(data):

    idx = np.flatnonzero(np.array([INTERCEPT_NAME == v for v in data['variable_names']]))

    if len(idx) > 0:

        assert len(idx) == 1, \
            "X has multiple columns named %s" % INTERCEPT_NAME

        assert np.all(data['X'][:, idx] == 1, axis = 0), \
            "found %s at column %d but X[:, %d] != 1.0" % (INTERCEPT_NAME, idx, idx)

        if 'X_test' in data and 'Y_test' in data:
            assert np.all(data['X_test'][:, idx] == 1, axis = 0), \
                "found %s at column %d but X_test[:, %d] != 1.0" % (INTERCEPT_NAME, idx, idx)

        if 'X_validation' in data and 'Y_validation' in data:
            assert np.all(data['X_validation'][:, idx] == 1, axis = 0), \
                "found %s at column %d but X_validation[:, %d] != 1.0" % (INTERCEPT_NAME, idx, idx)

    return True


def add_intercept(data, idx = INTERCEPT_IDX):

    if not has_intercept(data):

        data = add_variable(data,
                            name = INTERCEPT_NAME,
                            idx = idx,
                            values = 1.00,
                            test_values = 1.00,
                            validation_values = 1.00)

        assert check_intercept(data)

    return data


def remove_intercept(data):
    if has_intercept(data):
        return remove_variable(data, INTERCEPT_NAME)
    else:
        return data


#### variable names and indices ####

def get_index_of(data, names):
    if type(names) is list:
        return list(map(lambda n: get_index_of(data, n), names))
    else:
        return data['variable_names'].index(names)


def get_variable_names(data, include_intercept = False):

    var_names = list(data['variable_names'])
    if not include_intercept and has_intercept(data):
        var_names.pop(get_intercept_index(data))

    return var_names


def get_variable_indices(data, include_intercept = False):
    var_names = get_variable_names(data, include_intercept)
    idx = np.array([data['variable_names'].index(n) for n in var_names], dtype = int)
    return idx


#### duplicate variables in X ####

def check_full_rank(data):

    variable_idx = get_variable_indices(data, include_intercept = False)
    n_variables = len(variable_idx)

    X = np.array(data['X'][:, variable_idx], dtype = np.float_)
    assert n_variables == np.linalg.matrix_rank(X), 'X contains redundant features'

    if has_test_set(data):
        X_test = np.array(data['X_test'][: variable_idx], dtype = np.float_)
        assert n_variables == np.linalg.matrix_rank(X_test), 'X_test contains redundant features'

    if has_validation_set(data):
        X_validation = np.array(data['X_validation'][: variable_idx], dtype = np.float_)
        assert n_variables == np.linalg.matrix_rank(X_validation), 'X_validation contains redundant features'

    return True


def drop_trivial_variables(data):
    """
    drops unusuable features (due to trivial features etc)
    :param data:
    :return:
    """
    trivial_idx = np.all(data['X'] == data['X'][0, :], axis=0)

    if any(trivial_idx):

        drop_idx = np.flatnonzero(trivial_idx)
        variables_to_drop = [data['variable_names'][k] for k in drop_idx]
        for var_name in variables_to_drop:
            data = remove_variable(data, var_name)

    return data


def list_duplicate_variables(X):
    """

    :param X:
    :return:
    """
    d = X.shape[1]
    duplicates = []
    for (j, k) in itertools.combinations(range(d), 2):
        if np.array_equal(X[:, j], X[:, k]):
            duplicates.append((j, k))

    return duplicates


#### count # of conflicted points

def get_common_row_indices(A, B):
    """
    A and B need to contain unique rows
    :param A: array
    :param B: array
    :return:
    """
    nrows, ncols = A.shape
    dtype = {
        'names': ['f{}'.format(i) for i in range(ncols)],
        'formats': ncols * [A.dtype]
        }

    common_rows = np.intersect1d(A.view(dtype), B.view(dtype))
    if common_rows.shape[0] == 0:
        common_idx = np.empty((0, 2), dtype = np.dtype('int'))
    else:
        # common_rows = common_rows.view(A.dtype).reshape(-1, ncols)
        a_rows_idx = np.flatnonzero(np.isin(A.view(dtype), common_rows))
        b_rows_idx = np.flatnonzero(np.isin(B.view(dtype), common_rows))
        common_idx = np.column_stack((a_rows_idx, b_rows_idx))

    return common_idx


def count_conflicted_points(data):
    """
    :param data: dict containing 'X', 'Y'
    :return: data-related parameters to create lower bounding LP
    """
    X, Y = data['X'], data['Y']
    pos_idx = Y == 1
    U_pos, n_counts_pos = np.unique(X[pos_idx], axis = 0, return_counts = True)
    U_neg, n_counts_neg = np.unique(X[~pos_idx], axis = 0, return_counts = True)
    conflict_idx = get_common_row_indices(U_pos, U_neg)
    n_conflict = np.sum([min(n_counts_pos[p], n_counts_neg[q]) for p, q in conflict_idx])
    return n_conflict

#### saving data to disk

def save_data(file_name, data, cvindices = None, overwrite = False, stratified = True, check_save = True):

    f = Path(file_name)
    if overwrite is False:
        if f.exists():
            raise IOError('file %s already exist on disk' % file_name)

    file_type = f.suffix.lower()[1:]
    assert file_type in PROCESSED_DATA_FILE_VALID_EXT, \
        'unsupported extension %s\nsupported extensions: %s' % (file_type, ", ".join(PROCESSED_DATA_FILE_VALID_EXT))

    assert check_data(data)
    if cvindices is not None:
        cvindices = validate_cvindices(cvindices, stratified)

    if file_type in PROCESSED_DATA_FILE_RDATA:
        saved_file_flag = _save_data_as_rdata(file_name, data, cvindices)

    elif file_type in PROCESSED_DATA_FILE_PICKLE:
        saved_file_flag = _save_data_as_pickle(file_name, data, cvindices)

    assert f.exists(), 'file %s not found' % file_name

    if check_save:
        loaded_data, loaded_cvindices = load_processed_data(file_name)
        assert np.all(loaded_data['X'] == data['X'])
        assert loaded_cvindices.keys() == cvindices.keys()

    return saved_file_flag


def _save_data_as_pickle(file_name, data, cvindices):

    import pickle
    data = set_defaults_for_data(data)

    file_contents = {
        'data': data,
        'cvindices': cvindices
        }

    with open(file_name, 'wb') as outfile:
        pickle.dump(file_contents, outfile, protocol = pickle.HIGHEST_PROTOCOL)

    return True


def _save_data_as_rdata(file_name, data, cvindices):

    import rpy2.robjects as rn
    from .rpy2_helper import r_assign, r_save_to_disk
    from rpy2.robjects import pandas2ri
    data = set_defaults_for_data(data)
    assert check_data(data)

    fields_to_save = ["format", "Y", "sample_weights", "outcome_name", "variable_names"]

    try:

        for k in fields_to_save:
            r_assign(data[k], k)

    except:

        from eqm.debug import ipsh
        ipsh()

    r_assign(cvindices, "cvindices")

    pandas2ri.activate()

    X_df = pd.DataFrame(data = data['X'])
    X_df.columns = data['variable_names']
    rn.r.assign('X', X_df)

    # test set
    has_test_set = ('X_test' in data) and ('Y_test' in data) and ('sample_weights_test' in data)
    if has_test_set:
        X_test_df = pd.DataFrame(data = data['X_test'])
        X_test_df.columns = data['variable_names']
        rn.r.assign('X_test', pandas2ri.py2ri(X_test_df))
        r_assign(data['Y_test'], 'Y_test')
        r_assign(data['sample_weights_test'], 'sample_weights_test')
    else:
        rn.reval(
                """
                X_test = matrix(data=NA, nrow = 0, ncol = ncol(X));
                Y_test = matrix(data=NA, nrow = 0, ncol = 1);
                sample_weights_test = matrix(data=1.0, nrow = 0, ncol = 1);
                """
                )

    pandas2ri.deactivate()
    variables_to_save = fields_to_save + ["cvindices", "X", "X_test", "Y_test", "sample_weights_test"]
    r_save_to_disk(file_name, variables_to_save)
    return True


#### loading data from disk

def load_processed_data(file_name):

    f = Path(file_name)
    if not f.exists():
        raise IOError('file %s not found' % file_name)

    file_type = f.suffix.lower()[1:]
    assert file_type in PROCESSED_DATA_FILE_VALID_EXT, \
        'unsupported file type; supported file types: %s' % ", ".join(PROCESSED_DATA_FILE_VALID_EXT)

    if f.suffix.lower().endswith('rdata'):
        data, cvindices = _load_processed_data_rdata(file_name)
    else:
        data, cvindices = _load_processed_data_pickle(file_name)

    assert check_data(data)
    data = set_defaults_for_data(data)
    cvindices = validate_cvindices(cvindices)
    return data, cvindices


def _load_processed_data_pickle(file_name):

    import pickle
    with open(file_name, 'rb') as infile:
        file_contents = pickle.load(infile)

    assert 'data' in file_contents
    assert 'cvindices' in file_contents
    return file_contents['data'], file_contents['cvindices']


def _load_processed_data_rdata(file_name):

    import rpy2.robjects as rn

    rn.reval("data = new.env(); load('%s', data)" % file_name)
    r_data = rn.r.data
    data_fields = list(rn.r.data.keys())

    loaded_data = dict()
    for xf, yf, sw in [('X', 'Y', 'sample_weights'),
                       ('X_test', 'Y_test', 'sample_weights_test'),
                       ('X_validation', 'Y_validation', 'sample_weights_validation')]:

        if xf in data_fields and yf in data_fields and len(np.array(r_data[yf])) > 0:

            loaded_data[yf] = np.array(r_data[yf]).flatten()
            loaded_data[yf][loaded_data[yf] == 0] = -1
            loaded_data[xf] = np.array(r_data[xf])

            if loaded_data[xf].shape[1] == len(loaded_data[yf]):
                loaded_data[xf] = np.transpose(loaded_data[xf])

            if sw in data_fields:
                loaded_data[sw] = np.array(r_data[sw]).flatten()

    if 'variable_names' in data_fields:
        loaded_data['variable_names'] = np.array(rn.r.data['variable_names']).tolist()
    elif 'X_headers' in data_fields:
        loaded_data['variable_names'] = np.array(rn.r.data['X_headers']).tolist()
    elif 'X_header' in data_fields:
        loaded_data['variable_names'] = np.array(rn.r.data['X_header']).tolist()

    if 'outcome_name' in data_fields:
        loaded_data['outcome_name'] = np.array(r_data['outcome_name'])[0]
    elif 'Y_headers' in data_fields:
        loaded_data['outcome_name'] = np.array(r_data['Y_headers'])[0]
    elif 'Y_header' in data_fields:
        loaded_data['outcome_name'] = np.array(r_data['Y_header'])[0]

    if 'format' in data_fields:
        loaded_data['format'] = np.array(r_data['format'])[0]

    if 'partitions' in data_fields:
        loaded_data['partitions'] = np.array(rn.r.data['partitions']).tolist()

    cvindices = _load_cvindices_from_rdata(file_name)
    data = set_defaults_for_data(loaded_data)
    return data, cvindices


#### loading cvindices from processed file on disk ####


def load_cvindices_from_disk(fold_file):
    """
    Reads cross-validation indices from various file types including:
        - CSV file containing with exactly N data points
        - RData file containing cvindices object
        - mat file containing cvindices object
    :param fold_file:
    :return: dictionary containing folds
    keys have the form
    """
    # load fold indices from disk
    f = Path(fold_file)
    if not f.exists():
        raise IOError('could not find fold file on disk: %s' % f)

    if f.suffix.lower().endswith('csv'):
        folds = pd.read_csv(fold_file, sep=',', header=None)
        folds = validate_folds(folds=folds)
        fold_id = to_fold_id(total_folds = max(folds), replicate_idx = 1)
        cvindices = {fold_id: folds}

    if f.suffix.lower().endswith('rdata'):
        cvindices = _load_cvindices_from_rdata(data_file=fold_file)

    cvindices = validate_cvindices(cvindices)
    return cvindices


def _load_folds_from_rdata(data_file, fold_id):
    """
    (internal) returns folds from RData file in the pipeline
    :param data_file:
    :param fold_id:
    :param inner_fold_id:
    :return:
    """

    fold_id = validate_fold_id(fold_id)
    r_variables = "data_file='%s'; fold_id='%s'" % (data_file, fold_id)

    import rpy2.robjects as rn
    from .rpy2_helper import r_clear

    if is_inner_fold_id(fold_id):
        r_cmd = """raw_data = new.env()
        load(data_file, envir=raw_data)
        folds = raw_data$cvindices[[fold_id]][,1]
        """
    else:
        r_cmd = """raw_data = new.env()
        load(data_file, envir=raw_data)
        folds = raw_data$cvindices[[substring(fold_id, 1, 3)]][, as.double(substr(fold_id, 5, 6))]
        """

    rn.reval(r_variables)
    rn.reval(r_cmd)
    folds = np.array(rn.r['folds'])
    folds = validate_folds(folds, fold_id)
    r_clear()
    return folds


def _load_cvindices_from_rdata(data_file):
    """
    (internal) cvindices object stored in a RData file in the pipeline
    :param data_file:
    """
    import rpy2.robjects as rn
    from .rpy2_helper import r_clear

    r_variables = "data_file = '%s'" % data_file
    r_cmd = """
    raw_data = new.env()
    load(data_file, envir=raw_data)
    all_fold_ids = names(raw_data$cvindices)
    list2env(raw_data$cvindices, globalenv())
    remove(raw_data, cvindices);
    """
    rn.reval(r_variables)
    rn.reval(r_cmd)
    all_fold_ids = np.array(rn.r['all_fold_ids'])

    cvindices = {}
    max_fold_value = 0
    for key in all_fold_ids:
        try:
            folds = np.array(rn.r[key]).flatten()
            max_fold_value = max(max_fold_value, np.max(folds))
            cvindices[key] = folds
        except Exception:
            warnings.warn('failed to load fold %s from file %s' % (key, data_file))
            pass

    #cast as unsigned integers to save storage space
    if max_fold_value < 255:
        storage_type = 'uint8'
    elif max_fold_value < 65535:
        storage_type = 'uint16'
    elif max_fold_value < 4294967295:
        storage_type = 'uint32'
    else:
        storage_type = 'uint64'

    for key in cvindices.keys():
        cvindices[key] = cvindices[key].astype(storage_type)

    #break down matrices to just folds
    all_keys = list(cvindices.keys())
    for key in all_keys:
        if key[0] == 'K' and len(key) == 3:
            fold_matrix = cvindices.pop(key)
            n_repeats = fold_matrix.shape[1]
            for r in range(n_repeats):
                folds = np.array(fold_matrix[:,r])
                folds = folds.flatten()
                folds = folds.astype(storage_type)
                fold_id = '%sN%02d' % (key, r)
                cvindices[fold_id] = folds

    # cleanup in the R environment just in case
    r_clear()
    return cvindices



#### oversampling

def oversample_minority_class(data, **kwargs):
    """
    randomly oversamples the minority class so that there are an equal number of samples with positive and negative points
    all points are replicated according to sample weights
    :param data: data object
    :param kwargs: inputs passed to imblearn.over_sampling.RandomOverSampler
    :return:
    """
    assert check_data(data)

    X = data['X']
    Y = data['Y']
    assert np.isin((-1, 1), Y).all()

    if has_sample_weights(data):
        W = data['sample_weights']
        W = np.floor(W / np.min(W)).astype(int)
        X = np.repeat(X, W, axis = 0)
        Y = np.repeat(Y, W, axis = 0)

    # determine minority label
    pos_idx = Y == 1
    if pos_idx.sum() < len(Y) / 2:
        minority_label, majority_label = 1, -1
    else:
        minority_label, majority_label = -1, 1

    # oversample minority class
    ros = RandomOverRandomOverSampler(ratio = 'all', **kwargs)
    Xs, Ys = ros.fit_sample(X, Y)

    # double check sampling
    assert len(Y) <= len(Ys)
    assert np.sum(Y == majority_label) == np.sum(Ys == majority_label)
    assert np.sum(Y == minority_label) <= np.sum(Ys == minority_label)
    assert np.isclose(np.mean(Ys == 1), 0.5)

    # update data object
    data['X'] = np.array(Xs)
    data['Y'] = np.array(Ys)
    data['sample_weights'] = np.ones_like(Ys)
    assert check_data(data)
    return data


#### compress data

def compress_data(data):

    assert check_data(data)

    U, x_to_u_idx, u_to_x_idx,  N = np.unique(data['X'], axis = 0, return_inverse = True, return_index = True, return_counts = True)
    n_points = U.shape[0]
    N_pos, N_neg = np.zeros(n_points), np.zeros(n_points)
    for k in range(n_points):
        y = data['Y'][np.isin(u_to_x_idx, k)]
        N_pos[k], N_neg[k] = np.sum(y == 1), np.sum(y == -1)

    assert np.all(N_pos + N_neg == N)

    compressed = {
        'U': U,
        'x_to_u_idx': x_to_u_idx,
        'u_to_x_idx': u_to_x_idx,
        'N_pos': N_pos,
        'N_neg': N_neg,
        }

    return compressed


