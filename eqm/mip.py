from eqm.data import *
from eqm.cplex_mip_helper import Cplex, SparsePair, add_variable, set_mip_time_limit, StatsCallback, get_mip_stats, add_mip_start
from eqm.classifier_helper import ClassificationModel
from eqm.debug import ipsh

def to_mip_data(data):

    # choose label field
    Y = data['Y'].astype('float')
    Y = Y[:, 0] if Y.ndim > 1 and Y.shape[1] > 1 else Y
    Y = np.array(Y).flatten()

    # X
    X = data['X'].astype('float')
    n_samples, n_variables = X.shape
    # X_shift = np.zeros(n_variables)
    # X_scale = np.ones(n_variables)

    # variables and names
    variable_names = data['variable_names']
    intercept_idx = [variable_names.index(INTERCEPT_NAME)]
    coefficient_idx = [i for i, n in enumerate(variable_names) if n != INTERCEPT_NAME]

    # compression
    pos_ind = Y == 1
    neg_ind = ~pos_ind
    n_samples_pos = np.sum(pos_ind)
    n_samples_neg = n_samples - n_samples_pos
    X_pos = X[pos_ind, ]
    X_neg = X[neg_ind, ]
    U_pos, x_pos_to_u_pos_idx, u_pos_to_x_pos_idx, n_counts_pos = np.unique(X_pos, axis = 0, return_index = True, return_inverse = True, return_counts = True)
    U_neg, x_neg_to_u_neg_idx, u_neg_to_x_neg_idx, n_counts_neg = np.unique(X_neg, axis = 0, return_index = True, return_inverse = True, return_counts = True)
    n_points_pos = U_pos.shape[0]
    n_points_neg = U_neg.shape[0]

    x_to_u_pos_idx = np.flatnonzero(pos_ind)[x_pos_to_u_pos_idx]
    x_to_u_neg_idx = np.flatnonzero(neg_ind)[x_neg_to_u_neg_idx]

    assert np.all(X[x_to_u_pos_idx,] == U_pos)
    assert np.all(X[x_to_u_neg_idx,] == U_neg)
    assert np.all(Y[x_to_u_pos_idx,] == 1)
    assert np.all(Y[x_to_u_neg_idx,] == -1)

    mip_data = {
        #
        'format': 'mip',
        #
        'variable_names': variable_names,
        'intercept_idx': intercept_idx,
        'coefficient_idx': coefficient_idx,
        'n_variables': n_variables,
        #
        # data points
        'U_pos': U_pos,
        'U_neg': U_neg,
        'conflicted_pairs': get_common_row_indices(U_pos, U_neg),
        #
        # counts
        'n_samples': n_samples,
        'n_samples_pos': n_samples_pos,
        'n_samples_neg': n_samples_neg,
        'n_counts_pos': n_counts_pos,
        'n_counts_neg': n_counts_neg,
        'n_points_pos': n_points_pos,
        'n_points_neg': n_points_neg,
        #
        # debugging parameters
        'Y': Y,
        'x_to_u_pos_idx': x_to_u_pos_idx,
        'x_to_u_neg_idx': x_to_u_neg_idx,
        'u_pos_to_x_pos_idx': u_pos_to_x_pos_idx,
        'u_neg_to_x_neg_idx': u_neg_to_x_neg_idx,
        }

    return mip_data



ERROR_CONSTRAINT_TYPES = {
    0, # custom Big-M constraints
    1, # CPLEX indicators to set z_i = 1 unless point i is correctly classified
    3, # CPLEX indicators to set z_i = 0 if point i is correctly classified and z_i = 1 if point i is incorrectly classified
    }

# returns a cplex MIP object + "info" a dictionary of meta-info that I need to verify it's solution
def build_mip(data, settings, var_name_fmt, **kwargs):
    # todo: edit this

    """
    :param data:
    :param settings:
    :param var_name_fmt:
    :return:
    --
    variable vector = [theta_pos, theta_neg, sign, mistakes_pos, mistakes_neg, loss_pos, loss_neg] ---
    --
    ----------------------------------------------------------------------------------------------------------------
    name                  length              type        description
    ----------------------------------------------------------------------------------------------------------------
    theta_pos:            d x 1               real        positive components of weight vector
    theta_neg:            d x 1               real        negative components of weight vector
    theta_sign:           d x 1               binary      sign of weight vector. theta_sign[j] = 1 -> theta_pos > 0; theta_sign[j] = 0  -> theta_neg = 0.
    mistakes_pos:         n_points_pos x 1    binary      mistake_pos[i] = 1 if mistake on i
    mistakes_neg:         n_points_neg x 1    binary      mistake_neg[i] = 1 if mistake on i
    """
    assert data['format'] == 'mip'
    if len(kwargs) > 0:
        settings.update(kwargs)

    # check error constraint types
    assert settings['error_constraint_type'] in ERROR_CONSTRAINT_TYPES
    error_constraint_type = int(settings['error_constraint_type'])

    # basic mip parameters
    w_pos, w_neg = float(settings['w_pos']), 1.0
    total_l1_norm = np.abs(settings['total_l1_norm']).astype(float)
    assert np.greater(total_l1_norm, 0.0)
    margin = np.abs(settings['margin']).astype(float)
    assert np.isfinite(margin)

    # lengths
    n_variables = len(data['variable_names'])
    n_points_pos, n_points_neg = data['n_points_pos'], data['n_points_neg']
    n_samples = data['n_samples']

    # coefficient bounds
    theta_ub = np.repeat(total_l1_norm, n_variables)
    theta_lb = np.repeat(-total_l1_norm, n_variables)

    # bounds on the number of total mistakes
    mistakes_total_min = 0
    mistakes_total_max = n_samples

    # create MIP
    mip = Cplex()
    mip.objective.set_sense(mip.objective.sense.minimize)
    vars = mip.variables
    cons = mip.linear_constraints

    # create variable names
    print_vnames = lambda vfmt, vcnt: list(map(lambda v: vfmt % v, range(vcnt)))
    names = {
        'theta_pos': print_vnames(var_name_fmt['theta_pos'], n_variables),
        'theta_neg': print_vnames(var_name_fmt['theta_neg'], n_variables),
        'theta_sign': print_vnames(var_name_fmt['theta_sign'], n_variables),
        'mistakes_pos': print_vnames(var_name_fmt['mistake_pos'], n_points_pos),
        'mistakes_neg': print_vnames(var_name_fmt['mistake_neg'], n_points_neg),
        'total_mistakes': 'total_mistakes',
        }

    # add variables to MIP
    add_variable(mip, name = names['theta_pos'], obj = 0.0, ub = theta_ub, lb = 0.0, vtype = vars.type.continuous)
    add_variable(mip, name = names['theta_neg'], obj = 0.0, ub = 0.0, lb = theta_lb, vtype = vars.type.continuous)
    add_variable(mip, name = names['theta_sign'], obj = 0.0, ub = 1.0, lb = 0.0, vtype = vars.type.binary)
    add_variable(mip, name = names['mistakes_pos'], obj = 0.0, ub = 1.0, lb = 0.0, vtype = vars.type.binary)
    add_variable(mip, name = names['mistakes_neg'], obj = 0.0, ub = 1.0, lb = 0.0, vtype = vars.type.binary)
    add_variable(mip, name = names['total_mistakes'], obj = 1.0, ub = mistakes_total_max, lb = mistakes_total_min, vtype = vars.type.integer)

    theta_names = names['theta_pos'] + names['theta_neg']

    # define auxiliary variable for total mistakes
    con_ind = [names['total_mistakes']] + names['mistakes_pos'] + names['mistakes_neg']
    con_val = [-1.0] + (w_pos * data['n_counts_pos']).tolist() + data['n_counts_neg'].tolist()
    cons.add(names = ['def_total_mistakes'],
             lin_expr = [SparsePair(ind = con_ind, val = con_val)],
             senses = ['E'],
             rhs = [0.0])


    # add constraint to cap L1 Norm of coefficients
    # sum_j(theta_pos[j] - theta_neg[j]) = total_l1_norm
    cons.add(names = ['L1_norm_limit'],
             lin_expr = [SparsePair(ind = theta_names, val = [1.0] * n_variables + [-1.0] * n_variables)],
             senses = ['E'],
             rhs = [total_l1_norm])

    # add constraint to ensure that coefficients must be positive or negative
    # without this constraint sometimes we have that theta_pos[j] > 0 and theta_neg[j] > 0
    for T_pos, T_neg, g, tp, tn in zip(theta_ub, theta_lb, names['theta_sign'], names['theta_pos'], names['theta_neg']):
        # if tp[j] > 0 then g[j] = 1
        # if tn[j] < 0 then g[j] = 0
        # T_pos[j] * g[j] >= tp[j]       >>>>   T_pos[j] * g[j] - tp[j] >= 0
        # T_neg[j] * (1 - g[j]) >= -tn[j] >>>>  T_neg[j] * g[j] - tn[j] <= T_neg[j]
        cons.add(names = ['set_%r_pos' % g, 'set_%r_neg' % g],
                 lin_expr = [SparsePair(ind = [g, tp], val = [abs(T_pos), -1.0]),
                             SparsePair(ind = [g, tn], val = [abs(T_neg), -1.0])],
                 senses = ['G', 'L'],
                 rhs = [0.0, abs(T_neg)])


    # constraint for conflicting pairs
    if error_constraint_type == 0:

        # Big M constraints to set mistakes
        M_pos = margin + (total_l1_norm * np.max(np.abs(data['U_pos']), axis = 1))
        M_neg = margin + (total_l1_norm * np.max(np.abs(data['U_neg']), axis = 1))
        M_pos = M_pos.reshape((n_points_pos, 1))
        M_neg = M_neg.reshape((n_points_neg, 1))
        assert np.greater(M_pos, 0.0).all() and np.greater(M_neg, 0.0).all()

        pos_vals = np.hstack((data['U_pos'], data['U_pos'], M_pos)).tolist()
        for zp, val in zip(names['mistakes_pos'], pos_vals):
            cons.add(names = ['def_%r' % zp],
                     lin_expr = [SparsePair(ind = theta_names + [zp], val = val)],
                     senses = ['G'],
                     rhs = [margin])

        neg_vals = np.hstack((-data['U_neg'], -data['U_neg'], M_neg)).tolist()
        for zn, val in zip(names['mistakes_neg'], neg_vals):
            cons.add(names = ['def_%r' % zn],
                     lin_expr = [SparsePair(ind = theta_names + [zn], val = val)],
                     senses = ['G'],
                     rhs = [margin])


    else:

        pos_vals = np.tile(data['U_pos'], 2).tolist()
        neg_vals = np.tile(data['U_neg'], 2).tolist()

        # add constraints to set the mistake indicators for y[i] = +1
        for zp, val in zip(names['mistakes_pos'], pos_vals):

            # "z[i] = 0" -> "score[i] >= margin_pos[i]" is active
            mip.indicator_constraints.add(name = 'def_%r_off' % zp,
                                          indvar = zp,
                                          complemented = 1,
                                          lin_expr = SparsePair(ind = theta_names, val = val),
                                          sense = 'G',
                                          rhs = abs(margin),
                                          indtype = error_constraint_type)

            # "z[i] = 1" -> "score[i] < margin_pos[i]" is active
            # mip.indicator_constraints.add(name = 'def_%r_on' % zp,
            #                               indvar = zp,
            #                               complemented = 0,
            #                               lin_expr = SparsePair(ind = theta_names, val = val),
            #                               sense = 'L',
            #                               rhs = abs(margin),
            #                               indtype = 1)


        # add constarints to set the mistake indicators for y[i] = -1
        for zn, val in zip(names['mistakes_neg'], neg_vals):

            # "z[i] = 0" -> "score[i] <= margin_neg[i]" is active
            mip.indicator_constraints.add(name = 'def_%r_off' % zn,
                                          indvar = zn,
                                          complemented = 1,
                                          lin_expr = SparsePair(ind = theta_names, val = val),
                                          sense = 'L',
                                          rhs = -abs(margin),
                                          indtype = error_constraint_type)

            # mip.indicator_constraints.add(name = 'def_%r_on' % zn,
            #                               indvar = zn,
            #                               complemented = 0,
            #                               lin_expr = SparsePair(ind = theta_names, val = val),
            #                               sense = 'G',
            #                               rhs = -abs(margin),
            #                               indtype = 1)



    # collect information to validate solution
    info = {
        'settings': dict(settings),
        'names': names,
        'variable_idx': {name: idx for idx, name in enumerate(data['variable_names'])},
        'coefficient_idx': vars.get_indices(theta_names),
        'w_pos': w_pos,
        'w_neg': w_neg,
        'total_l1_norm': total_l1_norm,
        'margin': margin,
        'n_samples': n_samples,
        'upper_bounds': {k: np.array(vars.get_upper_bounds(n)) for k, n in names.items()},
        'lower_bounds': {k: np.array(vars.get_lower_bounds(n)) for k, n in names.items()},
        }

    return mip, info

# checks the solution of a MIP object
def check_mip_solution(mip, info, data, debug_flag = False):
    """
    implements basic tests to check that the 0-1 MIP is working correctly
    :param mip: MIP object produced by build_mimp
    :param info: info dict produced by "build_disc_mip" (this contains MIP level information that is used to check correctness)
    :param data: data
    :param debug_flag: set to True to call a debugger whenever a problem is encountered
    :return:
    """

    if debug_flag:
        from eqm.debug import ipsh as debugger
    else:
        debugger = lambda: True

    # check solutions
    if not mip.solution.is_primal_feasible():
        warnings.warn('no solution exists!')
        debugger()
        return True

    names = info['names']
    sol = mip.solution
    objval = sol.get_objective_value()
    theta_pos = np.array(sol.get_values(names['theta_pos']))
    theta_neg = np.array(sol.get_values(names['theta_neg']))
    theta_sign = np.array(sol.get_values(names['theta_sign']))
    mistakes_pos = np.array(sol.get_values(names['mistakes_pos']))
    mistakes_neg = np.array(sol.get_values(names['mistakes_neg']))
    theta = theta_pos + theta_neg

    try:
        # check objective value
        objval_check = data['n_counts_pos'].dot(mistakes_pos) + data['n_counts_neg'].dot(mistakes_neg)
        assert np.isclose(objval, objval_check)
    except AssertionError as e:
        print('objective error', e)
        debugger()


    try:
        # check coefficients
        assert set(np.flatnonzero(theta_pos)).isdisjoint(set(np.flatnonzero(theta_neg)))
        assert np.greater_equal(theta_pos, 0.0).all()
        assert np.less_equal(theta_neg, 0.0).all()
        assert np.isclose(info['total_l1_norm'], np.abs(theta).sum())
        assert np.isclose(theta_sign[theta > 0.0], 1.0).all()
        assert np.isclose(theta_sign[theta < 0.0], 0.0).all()
    except AssertionError as e:
        print('error in coefficient variable behavior')
        debugger()

    try:
        # check conflicted points
        conflicted_pairs = tuple(data['conflicted_pairs'])
        conflicted_mistakes = np.array([mistakes_pos[pi] + mistakes_neg[qi] for pi, qi in conflicted_pairs])
        assert np.all(conflicted_mistakes >= 1)
        if np.any(conflicted_mistakes >= 2):
            warnings.warn("BUG: found %d cases where z[i] + z[i'] >= 2 on conflicting points" % np.sum(conflicted_mistakes >= 2))

        # check lower bound on mistakes
        n_equivalent_points = sum([min(data['n_counts_pos'][pi], data['n_counts_neg'][qi]) for pi, qi in conflicted_pairs])
        assert np.greater_equal(objval, n_equivalent_points)

    except AssertionError as e:
        print('error with # of equivalent points', e)
        debugger()

    # ipsh()
    # idx = data['conflicted_pairs'][np.flatnonzero(conflicted_mistakes == 2)]
    # mistakes_pos[bug_idx[:, 0]]
    # mistakes_neg[bug_idx[:, 1]]
    # P = data['U_pos'][bug_idx[:, 0]]
    # Q = data['U_neg'][bug_idx[:, 1]]
    # assert np.all(P == Q)
    # Q.dot(theta)
    # P.dot(theta)
    # data['U_pos'][bug_idx[0, 0]] == data['U_neg'][bug_idx[0, 1]]


    # check that mistakes are actually miskakes
    y_pos = data['Y'][data['x_to_u_pos_idx']]
    y_neg = data['Y'][data['x_to_u_neg_idx']]
    assert np.all(y_pos == 1) and np.all(y_neg == -1)
    margin = np.abs(info['margin'])

    # compute scores and predictions for positive/negative points
    S_pos = data['U_pos'].dot(theta)
    S_neg = data['U_neg'].dot(theta)

    # compute mistakes for positive negative points
    expected_mistakes_pos = np.less_equal(S_pos, margin)
    expected_mistakes_neg = np.greater_equal(S_neg, -margin)

    # throw error if z[i] == 0 but s[i] >= margin
    bug_idx = np.flatnonzero((expected_mistakes_pos == 1) & (mistakes_pos == 0))
    bug_idx = np.setdiff1d(bug_idx, np.flatnonzero(np.isclose(S_pos, margin)))
    try:
        assert len(bug_idx) == 0
    except AssertionError as e:
        print('BUG: score[i] <  margin but mistakes_pos[i] = 0 for %1.0f points' % len(bug_idx))
        debugger()

    # throw error if z[i] == 0 but s[i] < margin
    bug_idx = np.flatnonzero((expected_mistakes_neg == 1) & (mistakes_neg == 0))
    bug_idx = np.setdiff1d(bug_idx, np.flatnonzero(np.isclose(S_neg, -margin)))
    try:
        assert len(bug_idx) == 0
    except AssertionError as e:
        print('BUG: score[i] >  -margin but mistakes_neg[i] = 0 for %1.0f points' % len(bug_idx))
        debugger()

    return True


def check_mip_solution_with_prediction_constraints(mip, info, data, prediction_constraints, debug_flag = False):
    """
    :param mip:
    :param info:
    :param data:
    :param debug_flag:
    :return:
    """

    # check solutions
    if not mip.solution.is_primal_feasible():
        warnings.warn('no solution exists!')
        return True

    if debug_flag:
        from eqm.debug import ipsh as debugger
    else:
        debugger = lambda: True

    # standard checks
    check_mip_solution(mip, info, data, debug_flag)

    # check for prediction constraints
    sol = mip.solution
    theta_pos = np.array(sol.get_values(info['names']['theta_pos']))
    theta_neg = np.array(sol.get_values(info['names']['theta_neg']))
    mistakes_pos = np.array(sol.get_values(info['names']['mistakes_pos']))
    mistakes_neg = np.array(sol.get_values(info['names']['mistakes_neg']))
    theta = theta_pos + theta_neg

    # extra tests for prediction constraint
    margin = np.abs(info['margin'])
    for x, yhat in prediction_constraints.values():
        score = np.dot(theta, x)

        # check that prediction constraint satisfied
        assert yhat in (-1, 1)
        if yhat == 1:
            try:
                assert np.greater_equal(score, margin)
            except AssertionError as e:
                print('BUG: MIP has constraint for h(x[i]) = +1, but score[i] < margin (%1.4f < %1.4f)' % (score, margin))
                debugger()

        elif yhat == -1:
            try:
                assert np.less_equal(score, -margin)
            except AssertionError as e:
                print('BUG: MIP has constraint for h(x[i]) = -1, but score[i] > -margin (%1.4f > %1.4f)' % (score, -margin))
                debugger()

        if np.not_equal(yhat, np.sign(score)):
            warnings.warn('MIP has constraint for h(x[i]) == %d, but sign(score) == %d' % (yhat, np.sign(score)))

        if yhat == 1:
            expected_mistakes_pos = 0
            expected_mistakes_neg = 1
        elif yhat == -1:
            expected_mistakes_pos = 1
            expected_mistakes_neg = 0

        # check that error is as expected on constrained point
        pos_idxs = np.flatnonzero((data['U_pos'] == x).all(axis = 1))
        neg_idxs = np.flatnonzero((data['U_neg'] == x).all(axis = 1))

        if any(pos_idxs):
            bugs_pos = np.not_equal(expected_mistakes_pos, mistakes_pos[pos_idxs]).sum()
            try:
                assert bugs_pos == 0
            except AssertionError as e:
                print('BUG: z[i] is false reporting when y[i] == -1 at %s points.' % bugs_pos)
                debugger()

        if any(neg_idxs):
            bugs_neg = np.not_equal(expected_mistakes_neg, mistakes_neg[neg_idxs]).sum()
            try:
                assert bugs_neg == 0
            except AssertionError as e:
                print('BUG: z[i] is false reporting when y[i] == +1 at %s points.' % bugs_neg)
                debugger()

    return True


class ZeroOneLossMIP(object):
    """
    Convenience class to create, solve, and check the integrity a CPLEX MIP
    that trains a linear classifier that minimizes the 0-1 loss
    """

    # default flags
    PRINT_FLAG = True
    PARALLEL_FLAG = True
    THREADS = 0

    # Default Parameters to the MIP
    SETTINGS = {
        'w_pos': 1.0,
        'margin': 0.0001,
        'error_constraint_type': 3,
        'total_l1_norm': 1.00,
        }

    # Default Formatters for the MIP Variables
    VAR_NAME_FMT = {
        'theta_pos': 'theta_pos_%d',
        'theta_neg': 'theta_neg_%d',
        'theta_sign': 'theta_sign_%d',
        'mistake_pos': 'zp_%d',
        'mistake_neg': 'zn_%d',
        }


    def __init__(self, data, print_flag = True, parallel_flag = False, random_seed = 2338, **kwargs):
        """
        :param data:
        :param print_flag:
        :param parallel_flag:
        """

        # check inputs
        assert isinstance(data, dict)
        assert isinstance(print_flag, bool)
        assert isinstance(parallel_flag, bool)

        # todo: maybe change
        self.original_data = data
        self.data = to_mip_data(data)
        cpx, info = build_mip(self.data, settings = self.SETTINGS, var_name_fmt = self.VAR_NAME_FMT, **kwargs)
        self.names = info['names']
        self.margin = float(info['margin'])
        self.random_seed = random_seed
        self._intercept_idx = np.array(self.data['intercept_idx'])
        self._coefficient_idx = np.array(self.data['coefficient_idx'])
        self.n_variables = len(self._coefficient_idx)

        # initialize prediction constraints
        self._prediction_constraints = {}

        # cplex parameters
        cpx = self._set_mip_parameters(cpx, random_seed)
        self.info = info

        checker_cpx, checker_info = build_mip(self.data, settings = self.SETTINGS, var_name_fmt = self.VAR_NAME_FMT)
        self._checker_cpx = self._set_mip_parameters(checker_cpx, random_seed)

        # easy access to CPLEX fields
        self.mip = cpx
        self.vars = self.mip.variables
        self.cons = self.mip.linear_constraints
        self.parameters = self.mip.parameters

        # indices
        indices = {}
        raw_indices = {k: self.vars.get_indices(v) for k, v in self.names.items()}
        for k, v in raw_indices.items():
            if isinstance(v, list):
                indices[k] = v
            else:
                indices[k] = [v]
        self._indices = indices

        # flags
        self.print_flag = print_flag
        self.parallel_flag = parallel_flag


    #### generic MIP functions ####

    def solve(self, time_limit = 60, node_limit = None, return_stats = False, return_incumbents = False):
        """
        solves MIP
        #
        :param time_limit: max # of seconds to run before stopping the B & B.
        :param node_limit: # of nodes to process in the B&B tree before stopping
        :param return_stats: set to True to record basic profiling information as the B&B algorithm runs (warning: this may slow down the B&B)
        :param return_incumbents: set to True to record all imcumbent solution processed during the B&B search (warning: this may slow down the B&B)
        :return:
        """
        attach_stats_callback = return_stats or return_incumbents

        if attach_stats_callback:
            self._add_stats_callback(store_solutions = return_incumbents)

        # update time limit
        if time_limit is not None:
            self.mip = set_mip_time_limit(self.mip, time_limit)

        if node_limit is not None:
            self.mip = set_mip_time_limit(self.mip, node_limit)

        # solve
        self.mip.solve()
        info = self.solution_info

        if attach_stats_callback:
            progress_info, progress_incumbents = self._stats_callback.get_stats()
            info.update({'progress_info': progress_info, 'progress_incumbents': progress_incumbents})

        return info


    def populate(self, max_gap = 0.0, time_limit = 60.0):
        """
        populates solution pool
        :param max_gap: set to 0.0 to find equivalent solutions
        :param time_limit:
        :return:
        """
        # generate populate stuff
        p = self.parameters
        p.mip.pool.replace.set(1)  # 1 = replace solutions with worst objective
        p.mip.pool.capacity.set(self.data['n_samples'])
        p.mip.pool.absgap.set(max_gap)  # gap for any equivalent solution should have gap = 0.0 (this doesn't seem to work)
        if max_gap == 0.0:
            p.mip.tolerances.absmipgap.set(0.0)  # remove the "tolerance" we had in place
        else:
            p.mip.tolerances.absmipgap.set(1.0)  # remove the "tolerance" we had in place
        p.timelimit.set(float(time_limit))  # solve for 60 seconds

        #call populate
        self.mip.populate_solution_pool()
        return True


    def check_solution(self, debug_flag = False):
        """
        runs basic tests to make sure that the MIP contains a suitable solution
        :return:
        """
        if len(self._prediction_constraints) > 0:
            assert check_mip_solution_with_prediction_constraints(mip = self.mip, info = self.info, data = self.data, prediction_constraints=self._prediction_constraints, debug_flag = debug_flag)
        else:
            assert check_mip_solution(mip = self.mip, info = self.info, data = self.data, debug_flag = debug_flag)


    @property
    def indices(self):
        return self._indices


    @property
    def print_flag(self):
        """
        set as True in order to print output information of the MIP
        :return:
        """
        return self._print_flag


    @print_flag.setter
    def print_flag(self, flag):
        if flag is None:
            self._print_flag = ZeroOneLossMIP.PRINT_FLAG
        elif isinstance(flag, bool):
            self._print_flag = bool(flag)
        else:
            raise ValueError('print_flag must be boolean or None')
        self._toggle_mip_display(cpx = self.mip, flag = self._print_flag)


    @property
    def parallel_flag(self):
        """
        set as True in order to print output information of the MIP
        :return:
        """
        return self._parallel_flag


    @parallel_flag.setter
    def parallel_flag(self, flag):
        if flag is None:
            self._parallel_flag = ZeroOneLossMIP.PARALLEL_FLAG
        elif isinstance(flag, bool):
            self._parallel_flag = bool(flag)
        else:
            raise ValueError('parallel_flag must be boolean or None')
        self._toggle_parallel(cpx = self.mip, flag = self._parallel_flag)


    @property
    def solution(self):
        """
        :return: handle to CPLEX solution
        """
        # todo add wrapper if solution does not exist
        return self.mip.solution


    @property
    def solution_info(self):
        """returns information associated with the current best solution for the mip"""
        return get_mip_stats(self.mip)


    #### generic ERM functions ####

    @property
    def coefficients(self):
        """
        :return: coefficients of the linear classifier
        """
        s = self.solution
        if s.is_primal_feasible():
            theta_pos = np.array(s.get_values(self.names['theta_pos']))
            theta_neg = np.array(s.get_values(self.names['theta_neg']))
            coefs = theta_pos + theta_neg
        else:
            coefs = np.repeat(np.nan, len(self.data['variable_names']))
        return coefs


    def solution_to_coefs(self, solution):
        """
        converts a solution to this MIP to a vector of coeffiients
        :param solution: solution vector from mip.solution or the solution pool
        :return: set of coefficients
        """
        theta = np.array([solution[j] for j in self.info['coefficient_idx']])
        d = len(theta) // 2
        assert len(theta) == (2 * d)
        return theta[0:d] + theta[d:]


    def get_classifier(self, coefs = None):
        """
        uses the coefficients to construct a classifier object similar to the classifiers in sklearn
        :return:
        """
        if coefs is None:
            coefs = self.coefficients
        else:
            assert len(coefs) == len(self.names)

        intercept_idx = np.array(self._intercept_idx)
        coefficient_idx = np.array(self._coefficient_idx)
        # if self._mip_settings['standardize_data']:
        #
        #     mu = np.array(self._mip_data['X_shift']).flatten()
        #     sigma = np.array(self._mip_data['X_scale']).flatten()
        #
        #     coefs = coefs / sigma
        #     total_shift = coefs.dot(mu)
        #
        #     coefficients = coefs[coefficient_idx]
        #
        #     if intercept_idx >= 0:
        #         intercept = coefs[intercept_idx] - total_shift
        #     else:
        #         intercept = -total_shift
        coefficients = coefs[self._coefficient_idx]
        if intercept_idx >= 0:
            intercept = coefs[self._intercept_idx]
        else:
            intercept = 0.0

        predict_handle = lambda X: np.sign(X[:, coefficient_idx].dot(coefficients) + intercept)

        # setup parameters for model object
        model_info = {
            'intercept': intercept,
            'coefficients': coefficients,
            'coefficient_idx': coefficient_idx,
            }

        training_info = {
            'method_name': 'mip',
            }

        model = ClassificationModel(predict_handle = predict_handle,
                                    model_type = ClassificationModel.LINEAR_MODEL_TYPE,
                                    model_info = model_info,
                                    training_info = training_info)

        return model


    #### prediction constraints

    def add_prediction_constraint(self, x, yhat, name = None):
        """
        add a prediction constraint to the MIP
        :param x: feature vector for the prediction constraint
        :param yhat: predicted class for x (must be either -1 or 1)
        :param name: name for the prediction class (optional)
        :return: True if prediction constraint added successfully.
        """
        # cast inputs
        x = np.array(x, dtype = float).flatten().tolist()
        yhat = int(yhat)
        assert len(x) == self.n_variables + 1
        assert isinstance(yhat, int) and yhat in (-1, 1)

        # make sure constraint does not already exist
        assert (x, yhat) not in self._prediction_constraints.values()
        if name is None:
            name_gen = lambda i: 'pred_constraint_{}'.format(i)
            name = next(name_gen(i) for i in range(int(1e10)) if name_gen(i) not in self._prediction_constraints.keys())
        assert isinstance(name, str) and name not in self._prediction_constraints.keys()

        # add constraint to cplex objects
        self.cons.add(names = [name],
                      lin_expr = [SparsePair(ind = self.names['theta_pos'] + self.names['theta_neg'], val = np.tile(x, 2).astype(float).tolist())],
                      senses = ['G'] if yhat == 1 else ['L'],
                      rhs = [yhat * self.margin])

        # store the constraint
        self._prediction_constraints[name] = (x, yhat)
        return True


    def clear_prediction_constraints(self, x = None, y = None, name = None):
        """
        clears prediction constraints
        :param x:
        :param y:
        :param name:
        :return:
        """
        if name is not None:
            assert isinstance(name, str)
            self.cons.delete(name)
            self._prediction_constraints.pop(name)
        elif x is not None and y in (-1, 1):
            name = None
            for c in self._prediction_constraints.items():
                _, xy = c
                if (x, y) == xy:
                    name = c[0]
            assert name is not None, 'could not find constraint matching (x = {}, y = {})'.format(x, y)
            self.cons.delete(name)
            self._prediction_constraints.pop(name)
        else:
            self.cons.delete(list(self._prediction_constraints.keys()))
            self._prediction_constraints.clear()

        return True

    #### helper functions

    def set_total_mistakes(self, lb = None, ub = None, strategy = 'indirect'):
        """
        :param lb:
        :param ub:
        :param strategy:
        :return:
        """

        # preconditions
        assert lb is None or isinstance(lb, (float, int))
        assert ub is None or isinstance(ub, (float, int))
        assert strategy in ('direct', 'indirect', 'both')

        if strategy in ('indirect', 'both'):
            if lb is not None:
                self.vars.set_lower_bounds(self.names['total_mistakes'], float(lb))
            else:
                self.vars.set_lower_bounds(self.names['total_mistakes'], 0.0)

            if ub is not None:
                self.vars.set_upper_bounds(self.names['total_mistakes'], float(ub))
            else:
                self.vars.set_upper_bounds(self.names['total_mistakes'], float(self.info['n_samples']))

        if strategy in ('direct', 'both'):
            p = self.parameters
            if lb is not None:
                p.mip.tolerances.lowercutoff.set(float(lb))
            else:
                p.mip.tolerances.lowercutoff.set(0.0)
            if ub is not None:
                p.mip.tolerances.uppercutoff.set(float(ub))
            else:
                p.mip.tolerances.uppercutoff.set(float(self.info['n_samples']))


    def add_mistake_cut(self, z = None):
        """
        :param z: mistake parameter
        :return:
        """

        mistake_names = self.names['mistakes_pos'] + self.names['mistakes_neg']
        if z is None:
            z = self.solution.get_values(mistake_names)
        assert len(z) == len(mistake_names)
        assert (np.isclose(z, 0.0) | np.isclose(z, 1.0)).all()
        z = np.round(z)
        S_on = np.isclose(z, 1.0).astype(float)
        S_off = np.isclose(z, 0.0).astype(float)
        con_ind = (S_on - S_off).tolist()
        con_rhs = len(z) - 1.0 - np.sum(S_off)

        self.cons.add(names=['eliminate_mistake'],
                      lin_expr=[SparsePair(ind = mistake_names, val = con_ind)],
                      senses=['L'],
                      rhs=[con_rhs])



    def add_initial_solution_with_coefficients(self, coefs):
        """
        converts coefficient vector of linear classifier to a partial solution for the MIP
        :param coefs:
        :return:
        """
        coefs = np.array(coefs).flatten()
        assert len(coefs) == (self.n_variables + 1)
        assert np.isfinite(coefs).all()
        coef_norm = np.abs(coefs).sum()
        if np.isclose(coef_norm, self.info['total_l1_norm']):
            coefs = self.info['total_l1_norm'] * coefs / coef_norm

        # add solution to MIP start pool
        sol = np.maximum(coefs, 0.0).tolist() + np.minimum(coefs, 0.0).tolist()
        idx = self.indices['theta_pos'] + self.indices['theta_neg']
        self.mip.MIP_starts.add(SparsePair(val = sol, ind = idx), self.mip.MIP_starts.effort_level.solve_MIP)


    def add_initial_solution(self, solution, objval = None, effort_level = 1, name = None, check_initialization = False):
        """
        :param solution: solution values to provide to the mip
        :param objval: objective value achieved by the solution. If provided, used to check the solution is accepted
        :param effort_level: integer describing the effort with which CPLEX will try to repair the solution
        :param name:
        :return: The new cpx object with the solution added.
        """
        self.mip = add_mip_start(self.mip, solution, effort_level, name)

        if check_initialization and objval is not None:
            current_flag = self.print_flag
            if current_flag:
                self.print_flag = False
            self.solve(time_limit = 1.0)
            cpx_objval = self.solution.get_objective_value()
            if current_flag:
                self.print_flag = True

            try:
                assert np.less_equal(cpx_objval, objval)
            except AssertionError:
                warnings.warn('initial solution did not improve current upperbound\nCPLEX objval: %1.2f\nexpected_objval')


    def enumerate_equivalent_solutions(self, pool, time_limit = 30):
        """
        :param pool:
        :param time_limit:
        :return:
        """

        # get mistake list
        mistake_names = self.names['mistakes_pos'] + self.names['mistakes_neg']
        z = self.solution.get_values(mistake_names)

        info = self.solution_info
        lb = info['lowerbound']
        objval = info['objval']

        # remove original solution from the equivalent models finder
        eq_mip = ZeroOneLossMIP(self.original_data, print_flag = self.print_flag, parallel_flag = self.parallel_flag, random_seed = self.random_seed)
        eq_mip.add_mistake_cut(z)
        eq_mip.set_total_mistakes(lb = lb)

        # search for equivalent models
        equivalent_output = []
        keep_looking = True
        while keep_looking:
            out = eq_mip.solve(time_limit = time_limit)
            if out['objval'] < objval:
                warnings.warn('found new solution with better objval than baseline')
                equivalent_output.append(out)
                eq_mip.add_mistake_cut()
            elif out['objval'] == objval:
                equivalent_output.append(out)
                eq_mip.add_mistake_cut()
            else:
                keep_looking = False
            pool.add_from_mip(eq_mip)

        return equivalent_output, pool


    #### generic MIP methods ####
    def _toggle_mip_display(self, cpx, flag):
        """
        toggles MIP display for CPLEX
        :param cpx: Cplex object
        :param flag: True to turn off MIP display
        :return:
        """
        p = cpx.parameters
        if flag:
            p.mip.display.set(p.mip.display.default())
            p.simplex.display.set(flag)
        else:
            p.mip.display.set(False)
            p.simplex.display.set(False)


    def _toggle_parallel(self, cpx, flag):
        """
        toggles parallelization in CPLEX
        :param cpx: Cplex object
        :param flag: True to turn off MIP display
        :return:
        """
        p = self.mip.parameters
        if flag:
            p.threads.set(0)
            p.parallel.set(0)
        else:
            p.parallel.set(1)
            p.threads.set(1)


    def _set_mip_parameters(self, cpx, random_seed):
        """
        sets CPLEX parameters
        :param cpx:
        :return:
        """
        p = cpx.parameters
        p.randomseed.set(random_seed)

        # annoyances
        p.paramdisplay.set(False)
        p.output.clonelog.set(0)

        p.mip.tolerances.mipgap.set(0.0)
        p.mip.tolerances.absmipgap.set(0.95)
        return cpx


    def _add_stats_callback(self, store_solutions = False):

        if not hasattr(self, '_stats_callback'):
            sol_idx = self.names['theta_pos'] + self.names['theta_neg']
            min_idx, max_idx = min(sol_idx), max(sol_idx)
            assert np.array_equal(np.array(sol_idx), np.arange(min_idx, max_idx + 1))
            cb = self.mip.register_callback(StatsCallback)
            cb.initialize(store_solutions, solution_start_idx = min_idx, solution_end_idx = max_idx)
            self._stats_callback = cb

