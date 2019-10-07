import numpy as np
import pandas as pd
from eqm.paths import *
from eqm.data import *
from eqm.mip import *
from eqm.tests.testing_helper_functions import *


pd.set_option('display.max_columns', 30)
np.set_printoptions(precision = 5)
np.set_printoptions(suppress = True)

#### Test Setup ####
data_name = 'mammo'
test_settings = {
    'margin': 0.0001,
    'total_l1_norm': 1.00,
    'add_l1_penalty': False,
    }
time_limit = 10
random_seed = 2338

## load / process data
data_file = '%s%s_binarized.csv' % (data_dir, data_name)
data = load_data_from_csv(data_file)

# solve MIP
mip = ZeroOneLossMIP(data = data, settings = test_settings)
mip.solve(time_limit = time_limit)
indices = mip.indices
mip_data = mip.data
mip_settings = mip.settings
mip_info = mip.mip_info


#### Setup Variables ####

# data related components
theta_pos = np.array(mip.solution.get_values(indices['theta_pos']))
theta_neg = np.array(mip.solution.get_values(indices['theta_neg']))
mistakes_pos = np.array(mip.solution.get_values(indices['mistakes_pos']), dtype = np.bool_)
mistakes_neg = np.array(mip.solution.get_values(indices['mistakes_neg']), dtype = np.bool_)
theta = theta_pos + theta_neg

# data related components
X = np.array(data['X'])
if mip_settings['standardize_data']:
    X = (X - mip_data['X_shift'])/(mip_data['X_scale'])

Y = mip_data['Y']
n_variables = X.shape[1]

U_pos = mip_data['U_pos']
x_to_u_pos_idx = mip_data['x_to_u_pos_idx']
u_neg_to_x_neg_idx = mip_data['u_neg_to_x_neg_idx']
n_counts_pos = mip_data['n_counts_pos']
n_points_pos = len(n_counts_pos)
y_pos = Y[x_to_u_pos_idx]

U_neg = mip_data['U_neg']
x_to_u_neg_idx = mip_data['x_to_u_neg_idx']
u_pos_to_x_pos_idx = mip_data['u_pos_to_x_pos_idx']
n_counts_neg = mip_data['n_counts_neg']
n_points_neg = len(n_counts_neg)
y_neg = Y[x_to_u_neg_idx]

conflicted_pairs = mip_data['conflicted_pairs']

# Expected Values
scores = X.dot(theta)
scores_pos = scores[x_to_u_pos_idx]
yhat_pos = np.sign(scores_pos)
expected_mistakes_pos = np.not_equal(yhat_pos, y_pos)

scores_neg = scores[x_to_u_neg_idx]
yhat_neg = np.sign(scores_neg)
expected_mistakes_neg = np.not_equal(yhat_neg, y_neg)

#### Data Tests

def test_compression_pos():
    pos_idx = data['Y'] == 1
    n_pos = np.sum(pos_idx)
    assert np.sum(np.sum(n_counts_pos) == n_pos)
    assert np.all(y_pos == 1.0)
    assert np.all(X[x_to_u_pos_idx,:] == U_pos)
    assert np.all(U_pos[u_pos_to_x_pos_idx,:] == X[pos_idx, :])


def test_compression_neg():
    neg_idx = data['Y'] == -1
    n_neg = np.sum(neg_idx)
    assert np.sum(np.sum(n_counts_neg) == n_neg)
    assert np.all(y_neg == -1.0)
    assert np.all(X[x_to_u_neg_idx,:] == U_neg)
    assert np.all(U_neg[u_neg_to_x_neg_idx,:] == X[neg_idx, :])


def test_conflicted_pairs():
    for (p, n) in tuple(conflicted_pairs):
        assert np.all(U_pos[p,:] == U_neg[n,:])
        assert y_pos[p] == 1
        assert y_neg[n] == -1


#### Basic MIP Tests

def test_mip_settings():
    mip_settings = mip.settings
    for k, v in test_settings.items():
        assert v == mip_settings[k], \
            'setting mismatch (%s): expected %r\n found %r' % (k, v, mip_settings[k])


def test_mip_data():

    expected_data = to_mip_data(data = data)
    for k, v in expected_data.items():
        assert np.all(v == mip_data[k])


def test_mip_intercept():

    intercept_idx = mip_data['intercept_idx']
    if test_settings['fit_intercept']:
        assert intercept_idx >= 0
        assert np.all(mip_data['U_pos'][:, intercept_idx] ==  1.0)
        assert np.all(mip_data['U_neg'][:, intercept_idx] ==  1.0)
        if has_intercept(data):
            expected_n_coefs = data['X'].shape[1]
        else:
            expected_n_coefs = data['X'].shape[1] + 1
    else:
        assert intercept_idx == -1
        if has_intercept(data):
            expected_n_coefs = data['X'].shape[1] - 1
        else:
            expected_n_coefs = data['X'].shape[1]

    assert len(theta) == expected_n_coefs


def test_mip_data_normalization():

    assert 'intercept_idx' in mip_data
    assert 'coefficient_idx' in mip_data
    assert 'X_shift' in mip_data
    assert 'X_scale' in mip_data

    mu = np.array(mip_data['X_shift']).flatten()
    sigma = np.array(mip_data['X_scale']).flatten()
    intercept_idx = mip_data['intercept_idx']
    coefficient_idx = mip_data['coefficient_idx']

    if intercept_idx > 0:
        assert mu[intercept_idx] == 0.0
        assert sigma[intercept_idx] == 1.0

    assert np.all(np.greater(sigma, 0.0))


def test_mip_variable_bounds():
    for var_name in mip_info['var_names']:
        vals = mip.solution.get_values(indices[var_name])
        assert np.all(mip_info['lower_bounds'][var_name] <= vals)
        assert np.all(mip_info['upper_bounds'][var_name] >= vals)


def test_mip_variable_types():
    for var_name in mip_info['var_names']:
        vals = mip.solution.get_values(indices[var_name])
        types = mip_info['var_types'][var_name]
        if isinstance(vals, list):
            assert np.all(np.array(list(map(lambda vt: check_variable_type(vt[0], vt[1]), zip(vals, types)))))
        else:
            assert check_variable_type(vals, types)


def test_mip_variable_indices():

    indices = mip.indices

    # check types
    assert isinstance(indices, dict)
    for k in indices.keys():
        assert isinstance(indices[k], list)


    # check lengths
    assert len(indices['theta_pos']) == n_variables
    assert len(indices['theta_neg']) == n_variables
    assert len(indices['mistakes_pos']) == n_points_pos
    assert len(indices['mistakes_neg']) == n_points_neg
    assert len(indices['total_mistakes_pos']) == 1
    assert len(indices['total_mistakes_neg']) == 1
    assert len(indices['total_mistakes']) == 1

    if mip_info['add_coefficient_sign_constraints']:
        assert 'theta_sign' in indices
        sign_idx = np.array(mip.solution.get_values(indices['theta_sign']), dtype = np.bool_)
        assert len(sign_idx) == n_variables

    flat_indices = [item for var_indices in indices.values() for item in var_indices]
    flat_indices = np.sort(flat_indices)

    assert np.array_equal(flat_indices, mip.variables.get_indices(mip.variables.get_names())), \
        'indices are not distinct'

    assert np.array_equal(flat_indices, np.arange(start = flat_indices[0], stop = flat_indices[-1] + 1)), \
        'indices are not consecutive'

    assert np.array_equal(flat_indices, mip.variables.get_indices(mip.variables.get_names())), \
        'indices do not contain all variables in MIP object'


#### Key Attributes

def test_coefficients():
    indices = mip.indices
    theta_pos = np.array(mip.solution.get_values(indices['theta_pos']))
    theta_neg = np.array(mip.solution.get_values(indices['theta_neg']))
    theta = theta_pos + theta_neg
    assert np.array_equal(theta, mip.coefficients())


def test_model():
    model = mip.get_classifier()
    assert isinstance(model, ClassificationModel)
    assert model.check_rep()
    predictions_model = model.predict(data['X'])
    predictions_mip = np.sign(X.dot(mip.coefficients()))
    assert np.array_equal(predictions_model, predictions_mip)


#### Tests on Coefficients


def test_theta_pos_or_theta_neg():
    # check that either theta_pos > 0 or theta_neg < 0 not both
    assert ~np.any(np.logical_and(theta_pos, theta_neg)), "theta_pos[j] != 0 and theta_neg[j] != 0"


def test_sign_indicators():
    if mip_info['add_coefficient_sign_constraints']:
        sign_idx = np.array(mip.solution.get_values(indices['theta_sign']), dtype = np.bool_)
        assert np.all(theta_pos[sign_idx] >= 0.0)
        assert np.all(theta_neg[sign_idx] <= 0.0)


def test_L1_norm():
    assert np.isclose(np.sum(abs(theta)), mip_info['total_l1_norm'])
    if np.sum(abs(theta)) != mip_info['total_l1_norm']:
        msg = [
            'numerical issue in L1-norm,'
            'expected value: %1.6f' % mip_info['total_l1_norm'],
            'computed value: %1.6f' % np.sum(abs(theta))
            ]
        print('\n'.join(msg))

#### Parameters for Indicator Constraints


def test_margin_pos():
    assert np.all(mip_info['margin_pos'] == abs(mip_settings['margin']))
    assert np.all(mip_info['margin_pos'] >= 0.0)


def test_margin_neg():
    assert np.all(mip_info['margin_neg'] == abs(mip_settings['margin']))
    assert np.all(mip_info['margin_neg'] >= 0.0)


#### Classifier Scores


def test_score_is_within_margin_pos():
    margin_idx = (scores_pos > 0.0) & (scores_pos < mip_info['margin_pos'])
    bug_idx = np.where(margin_idx)[0]
    error_msg = "score is within margin for %d points\n" % len(bug_idx)
    error_msg += print_score_error(bug_idx, scores_pos, theta, U_pos, y_pos, mistakes_pos, point_type = "pos")


def test_score_is_within_margin_neg():
    margin_idx = (scores_neg > 0.0) & (scores_neg < mip_info['margin_neg'])
    bug_idx = np.where(margin_idx)[0]
    error_msg = "score is within margin for %d points\n" % len(bug_idx)
    error_msg += print_score_error(bug_idx, scores_pos, theta, U_pos, y_pos, mistakes_pos, point_type = "pos")


def test_score_is_exactly_zero_pos():
    bug_idx = np.where(scores_pos == 0.0)[0]
    error_msg = "score is exactly = 0.0 for %d points\n" % len(bug_idx)
    error_msg += print_score_error(bug_idx, scores_pos, theta, U_pos, y_pos, mistakes_pos, point_type = "pos")
    assert len(bug_idx) == 0, error_msg


def test_score_is_exactly_zero_neg():
    bug_idx = np.where(scores_neg == 0.0)[0]
    error_msg = "score is exactly = 0.0 for %d points\n" % len(bug_idx)
    error_msg += print_score_error(bug_idx, scores_neg, theta, U_neg, y_neg, mistakes_neg, point_type = "neg")
    assert len(bug_idx) == 0, error_msg


def test_score_is_small_pos():
    exact_idx = np.where(scores_pos == 0.0)[0]
    close_idx = np.flatnonzero(np.isclose(scores_pos, 0.0, rtol = 1e-10, atol = 1e-10))
    bug_idx = np.setdiff1d(close_idx, exact_idx, assume_unique = True)

    error_msg = "score is close to 0.0 for %d points\n" % len(bug_idx)
    error_msg += print_score_error(bug_idx, scores_pos, theta, U_pos, y_pos, mistakes_pos, point_type = "pos")
    assert len(bug_idx) == 0, error_msg


def test_score_is_small_neg():
    exact_idx = np.where(scores_neg == 0.0)[0]
    close_idx = np.flatnonzero(np.isclose(scores_neg, 0.0, rtol = 1e-10, atol = 1e-10))
    bug_idx = np.setdiff1d(close_idx, exact_idx, assume_unique = True)

    error_msg = "score is close to 0.0 for %d points\n" % len(bug_idx)
    error_msg += print_score_error(bug_idx, scores_neg, theta, U_neg, y_neg, mistakes_neg, point_type = "pos")
    assert len(bug_idx) == 0, error_msg

#### Mistakes

def test_mistakes_pos():
    bug_idx = np.flatnonzero(np.not_equal(mistakes_pos, expected_mistakes_pos))
    error_msg = "incorrect mistake variables for %d points\n" % len(bug_idx)
    error_msg += print_score_error(bug_idx, scores_pos, theta, U_pos, y_pos, mistakes_pos, point_type = "pos")
    assert len(bug_idx) == 0, error_msg


def test_mistakes_neg():
    bug_idx = np.flatnonzero(np.not_equal(mistakes_neg, expected_mistakes_neg))
    error_msg = "incorrect mistake variables for %d points\n" % len(bug_idx)
    error_msg += print_score_error(bug_idx, scores_neg, theta, U_neg, y_neg, mistakes_neg, point_type = "neg")
    assert len(bug_idx) == 0, error_msg


def test_mistakes_on_conflicts():
    for (p, n) in tuple(conflicted_pairs):
        error_msg = "found conflicting pair such that (l_pos[%d], l_neg[%d]) = (%d, %d)" % \
                    (p,n, mistakes_pos[p], mistakes_neg[n])
        assert mistakes_pos[p] + mistakes_neg[n] == 1, error_msg
