import numpy as np
from eqm.data import add_intercept,  has_intercept, check_intercept,load_processed_data
from eqm.cross_validation import filter_data_to_fold

# print options
np.set_printoptions(precision = 1, suppress = False)

#### Functions

def is_integer(x):
    return np.array_equal(x, np.require(x, dtype=np.int_))


def is_binary(x):
    return np.array_equal(x, np.require(x, dtype=np.bool_))


def check_variable_type(val, type):
    if type == 'B':
        return is_binary(val)
    elif type == 'I':
        return is_integer(val)
    elif type == 'C':
        return np.isfinite(val)
    else:
        return False


def compare_with_incumbent_solution(cpx, solution, indices = None):

    if indices is None:
        expected_solution = np.array(cpx.solution.get_values())
    else:
        expected_solution = np.array(cpx.solution.get_values(indices))

    diff_idx = np.flatnonzero(solution != expected_solution)

    if len(diff_idx) > 0:
        diff_names = cpx.variables.get_names(diff_idx.tolist())
    else:
        diff_names = []

    return diff_idx, diff_names


def print_score_error(bug_idx, score_values, theta, U, Y, L, point_type = "pos"):

    error_msg = ""
    n_variables = len(theta)

    for i in bug_idx:

        row_msg = "\nU_%s[%d,:] \t=\t(" % (point_type, i)
        for j in range(n_variables):
            if j < (n_variables - 1):
                row_msg += "%1.5f, " % U[i, j]
            else:
                row_msg += "%1.5f)\n" % U[i, j]

        row_msg += "theta \t\t=\t(" % i
        for j in range(n_variables):
            if j < (n_variables - 1):
                row_msg += "%1.5f, " % theta[j]
            else:
                row_msg += "%1.5f)\n" % theta[j]

        row_msg += "score[%d]  \t=\t%1.5f\n\n" % (i, score_values[i])
        row_msg += "y_hat[%d]  \t=\t%d\n" % (i, np.sign(score_values[i]))
        row_msg += "y_true[%d] \t=\t%d\n" % (i, Y[i])
        row_msg += "l_true[%d] \t=\t%d\n" % (i, Y[i] != np.sign(score_values[i]))
        row_msg += "l_cplex[%d] \t=\t%d\n\n" % (i, L[i])

        error_msg += row_msg

    return error_msg

