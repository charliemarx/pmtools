from eqm.data import *
from eqm.cplex_mip_helper import Cplex, SparsePair, add_variable, set_mip_time_limit, StatsCallback, get_mip_stats, add_mip_start
from eqm.classifier_helper import ClassificationModel
from eqm.debug import ipsh

ERROR_CONSTRAINT_TYPES = {
    0, # custom Big-M constraints
    1, # CPLEX indicators to set z_i = 1 unless point i is correctly classified
    3, # CPLEX indicators to set z_i = 0 if point i is correctly classified and z_i = 1 if point i is incorrectly classified
    }

def to_disc_mip_data(data, baseline_coefs):

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

    # B
    baseline_coefs = np.array(baseline_coefs).flatten()
    assert len(baseline_coefs) == n_variables
    B = np.sign(data['X'].dot(baseline_coefs))
    assert np.isin(B, (-1, 1)).all()

    # compression
    pos_ind = B == 1
    neg_ind = ~pos_ind
    n_samples_pos = np.sum(pos_ind)
    n_samples_neg = n_samples - n_samples_pos
    XY = np.block([Y[:, None], X])
    X_pos = XY[pos_ind, ]
    X_neg = XY[neg_ind, ]
    U_pos, x_pos_to_u_pos_idx, u_pos_to_x_pos_idx, n_counts_pos = np.unique(X_pos, axis = 0, return_index = True, return_inverse = True, return_counts = True)
    U_neg, x_neg_to_u_neg_idx, u_neg_to_x_neg_idx, n_counts_neg = np.unique(X_neg, axis = 0, return_index = True, return_inverse = True, return_counts = True)
    x_to_u_pos_idx = np.flatnonzero(pos_ind)[x_pos_to_u_pos_idx]
    x_to_u_neg_idx = np.flatnonzero(neg_ind)[x_neg_to_u_neg_idx]

    # basic assesstions
    assert np.all(X_pos[x_pos_to_u_pos_idx,] == U_pos)
    assert np.all(X_neg[x_neg_to_u_neg_idx,] == U_neg)
    assert np.all(XY[x_to_u_pos_idx, ] == U_pos)
    assert np.all(XY[x_to_u_neg_idx, ] == U_neg)
    assert np.all(B[x_to_u_pos_idx] == 1)
    assert np.all(B[x_to_u_neg_idx] == -1)
    assert len(get_common_row_indices(U_pos, U_neg)) == 0

    # compute loss indices for baseline classifier
    L = 1 - 2.0 * np.not_equal(B, Y)
    assert np.isin(L, (-1, 1)).all()
    LU_pos = L[x_to_u_pos_idx]
    LU_neg = L[x_to_u_neg_idx]
    assert np.not_equal(Y, B).sum() == np.sum(n_counts_pos[np.less(LU_pos, 0)]) + np.sum(n_counts_neg[np.less(LU_neg, 0)])

    mip_data = {
        #
        'format': 'disc_mip',
        #
        'variable_names': variable_names,
        'intercept_idx': intercept_idx,
        'coefficient_idx': coefficient_idx,
        'n_variables': n_variables,
        #
        # data points
        'U_pos': U_pos[:, 1:],
        'U_neg': U_neg[:, 1:],
        #
        'LU_pos': LU_pos,
        'LU_neg': LU_neg,
        #
        'conflicted_pairs': [],
        #
        # counts
        'n_samples': n_samples,
        'n_samples_pos': n_samples_pos,
        'n_samples_neg': n_samples_neg,
        'n_counts_pos': n_counts_pos,
        'n_counts_neg': n_counts_neg,
        'n_points_pos': U_pos.shape[0],
        'n_points_neg': U_neg.shape[0],
        #
        # debugging parameters
        'Y': Y,
        'B': B,
        'L': L,
        'x_to_u_pos_idx': x_to_u_pos_idx,
        'x_to_u_neg_idx': x_to_u_neg_idx,
        'u_pos_to_x_pos_idx': u_pos_to_x_pos_idx,
        'u_neg_to_x_neg_idx': u_neg_to_x_neg_idx,
        }



    return mip_data


def build_disc_mip(data, baseline_coefs, settings, var_name_fmt, **kwargs):

    """
    :param data:
    :param settings:
    :param baseline_coefs:
    :param var_name_fmt:
    :return:
    --
    variable vector = [theta_pos, theta_neg, sign, mistakes_pos, agree_neg, loss_pos, loss_neg] ---
    --
    ----------------------------------------------------------------------------------------------------------------
    name                  length              type        description
    ----------------------------------------------------------------------------------------------------------------
    theta_pos:            d x 1               real        positive components of weight vector
    theta_neg:            d x 1               real        negative components of weight vector
    theta_sign:           d x 1               binary      sign of weight vector. theta_sign[j] = 1 -> theta_pos > 0; theta_sign[j] = 0  -> theta_neg = 0.
    agree_pos:         n_points_pos x 1       binary      agree_pos[i] = 1 if h(x) = h0(x) for h0(x) = 1
    agree_neg:         n_points_neg x 1       binary      agree_neg[i] = 1 if h(x) = h0(x) for h0(x) = -1
    """
    assert data['format'] == 'disc_mip'
    if len(kwargs) > 0:
        settings.update(kwargs)

    # check baseline coefs
    baseline_coefs = np.array(baseline_coefs).flatten()
    assert len(baseline_coefs) == data['U_pos'].shape[1]
    assert np.isfinite(baseline_coefs).all()

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

    # create MIP
    mip = Cplex()
    mip.objective.set_sense(mip.objective.sense.minimize)
    vars = mip.variables
    cons = mip.linear_constraints

    # create variable names
    print_vnames = lambda vfmt, vcnt: list(map(lambda v: vfmt % v, range(vcnt)))
    names = {
        'total_agreement': 'a',
        'total_error_gap': 'g',
        'agree_pos': print_vnames(var_name_fmt['agree_pos'], n_points_pos),
        'agree_neg': print_vnames(var_name_fmt['agree_neg'], n_points_neg),
        #
        'theta_pos': print_vnames(var_name_fmt['theta_pos'], n_variables),
        'theta_neg': print_vnames(var_name_fmt['theta_neg'], n_variables),
        'theta_sign': print_vnames(var_name_fmt['theta_sign'], n_variables),
        }


    add_variable(mip, name = names['total_agreement'], obj = 1.0, ub = n_samples, lb = 0.0, vtype = vars.type.integer)
    add_variable(mip, name = names['total_error_gap'], obj = 0.0, ub = n_samples, lb = 0.0, vtype = vars.type.integer)
    add_variable(mip, name = names['agree_pos'], obj = 0.0, ub = 1.0, lb = 0.0, vtype = vars.type.binary)
    add_variable(mip, name = names['agree_neg'], obj = 0.0, ub = 1.0, lb = 0.0, vtype = vars.type.binary)

    # add variables to MIP
    add_variable(mip, name = names['theta_pos'], obj = 0.0, ub = theta_ub, lb = 0.0, vtype = vars.type.continuous)
    add_variable(mip, name = names['theta_neg'], obj = 0.0, ub = 0.0, lb = theta_lb, vtype = vars.type.continuous)
    add_variable(mip, name = names['theta_sign'], obj = 0.0, ub = 1.0, lb = 0.0, vtype = vars.type.binary)
    theta_names = names['theta_pos'] + names['theta_neg']

    # define variable for agreement
    con_ind = [names['total_agreement']] + names['agree_pos'] + names['agree_neg']
    con_val = [-1.0] + data['n_counts_pos'].tolist() + data['n_counts_neg'].tolist()
    cons.add(names = ['def_total_agreement'],
             lin_expr = [SparsePair(ind = con_ind, val = con_val)],
             senses = ['E'],
             rhs = [0.0])

    # define variable for discrepancy
    con_ind = [names['total_error_gap']] + names['agree_pos'] + names['agree_neg']
    count_factor_pos = np.multiply(data['LU_pos'], data['n_counts_pos'])
    count_factor_neg = np.multiply(data['LU_neg'], data['n_counts_neg'])
    con_val = [1.0] + count_factor_pos.tolist() + count_factor_neg.tolist()
    con_rhs = count_factor_pos.sum() + count_factor_neg.sum()
    cons.add(names = ['def_total_error_gap'],
             lin_expr = [SparsePair(ind = con_ind, val = con_val)],
             senses = ['E'],
             rhs = [float(con_rhs)])

    # add constraint to cap L1 norm of coefficients
    # sum_j(theta_pos[j] - theta_neg[j]) = total_l1_norm
    cons.add(names = ['L1_norm_limit'],
             lin_expr = [SparsePair(ind = theta_names, val = [1.0] * n_variables + [-1.0] * n_variables)],
             senses = ['E'],
             rhs = [total_l1_norm])

    # add constraint to ensure that coefficients must be positive or negative
    # without this constraint we may have that theta_pos[j] > 0 and theta_neg[j] > 0
    for T_pos, T_neg, g, tp, tn in zip(theta_ub, theta_lb, names['theta_sign'], names['theta_pos'], names['theta_neg']):
        cons.add(names = ['set_%r_pos' % g, 'set_%r_neg' % g],
                 lin_expr = [SparsePair(ind = [g, tp], val = [abs(T_pos), -1.0]), SparsePair(ind = [g, tn], val = [abs(T_neg), -1.0])],
                 senses = ['G', 'L'],
                 rhs = [0.0, abs(T_neg)])

    # constraint for conflicting pairs
    if error_constraint_type > 0:

        pos_vals = np.tile(data['U_pos'], 2).tolist()
        neg_vals = np.tile(data['U_neg'], 2).tolist()

        # add constraints to set the mistake indicators for y[i] = +1
        for ap, val in zip(names['agree_pos'], pos_vals):

            # "alpha[i] = 1" <-> "score[i] >= margin_pos[i]" is active
            mip.indicator_constraints.add(name = 'def_%r_off' % ap,
                                          indvar = ap,
                                          complemented = 0,
                                          lin_expr = SparsePair(ind = theta_names, val = val),
                                          sense = 'G',
                                          rhs = abs(margin),
                                          indtype = 3)

        # add constarints to set the mistake indicators for y[i] = -1
        for an, val in zip(names['agree_neg'], neg_vals):

            # "alpha[i] = 1" <-> "score[i] < -margin_neg[i]" is active
            mip.indicator_constraints.add(name = 'def_%r_off' % an,
                                          indvar = an,
                                          complemented = 0,
                                          lin_expr = SparsePair(ind = theta_names, val = val),
                                          sense = 'L',
                                          rhs = -abs(margin),
                                          indtype = 3)


    else:

        # Big M constraints to set mistakes
        M_pos = margin + (total_l1_norm * np.max(np.abs(data['U_pos']), axis = 1))
        M_neg = margin + (total_l1_norm * np.max(np.abs(data['U_neg']), axis = 1))
        M_pos = M_pos.reshape((n_points_pos, 1))
        M_neg = M_neg.reshape((n_points_neg, 1))
        assert np.greater(M_pos, 0.0).all() and np.greater(M_neg, 0.0).all()

        pos_vals = np.hstack((data['U_pos'], data['U_pos'], M_pos)).tolist()
        for zp, val in zip(names['agree_pos'], pos_vals):
            cons.add(names = ['def_%r' % zp],
                     lin_expr = [SparsePair(ind = theta_names + [zp], val = val)],
                     senses = ['G'],
                     rhs = [margin])

        neg_vals = np.hstack((-data['U_neg'], -data['U_neg'], M_neg)).tolist()
        for zn, val in zip(names['agree_neg'], neg_vals):
            cons.add(names = ['def_%r' % zn],
                     lin_expr = [SparsePair(ind = theta_names + [zn], val = val)],
                     senses = ['G'],
                     rhs = [margin])

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



class DiscrepancyMIP(object):
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
        'margin': 1e-6,
        'error_constraint_type': 3,
        'total_l1_norm': 1.00,
        }

    # Default Formatters for the MIP Variables
    VAR_NAME_FMT = {
        'agree_pos': 'ap_%d',
        'agree_neg': 'an_%d',
        'theta_pos': 'theta_pos_%d',
        'theta_neg': 'theta_neg_%d',
        'theta_sign': 'theta_sign_%d',
        }


    def __init__(self, data, baseline_coefs, baseline_stats, print_flag = True, parallel_flag = False, random_seed = 2338, **kwargs):
        """
        :param data:
        :param print_flag:
        :param parallel_flag:
        """

        # check inputs
        assert isinstance(data, dict)
        assert isinstance(print_flag, bool)
        assert isinstance(parallel_flag, bool)
        assert isinstance(baseline_coefs, (list, np.ndarray))
        assert isinstance(baseline_stats, dict)

        # attach baseline things
        self.baseline_coefs = np.array(baseline_coefs).flatten()
        self.baseline_stats = {k: v for (k, v) in baseline_stats.items()}

        # attach data
        self.original_data = data
        self.data = to_disc_mip_data(data, baseline_coefs)

        # setup mip
        cpx, info = build_disc_mip(self.data, baseline_coefs, settings = self.SETTINGS, var_name_fmt = self.VAR_NAME_FMT, **kwargs)
        cpx = self._set_mip_parameters(cpx, random_seed)
        #
        self.names = info['names']
        self.margin = float(info['margin'])
        self.random_seed = random_seed
        self._intercept_idx = np.array(self.data['intercept_idx'])
        self._coefficient_idx = np.array(self.data['coefficient_idx'])
        self.n_variables = len(self._coefficient_idx)



        # set checker cplex
        checker_cpx, checker_info = build_disc_mip(self.data, baseline_coefs = self.baseline_coefs, settings = self.SETTINGS, var_name_fmt = self.VAR_NAME_FMT)
        self._checker_cpx = self._set_mip_parameters(checker_cpx, random_seed)

        # attach CPLEX object
        self.mip = cpx
        self.vars = self.mip.variables
        self.cons = self.mip.linear_constraints
        self.parameters = self.mip.parameters
        self.info = info

        # indices
        indices = {}
        raw_indices = {k: self.vars.get_indices(v) for k, v in self.names.items()}
        for k, v in raw_indices.items():
            if isinstance(v, list):
                indices[k] = v
            else:
                indices[k] = [v]
        self._indices = indices

        # add initial solution
        self.add_initial_solution_with_coefficients(coefs = baseline_coefs)

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

        if debug_flag:
            from eqm.debug import ipsh as debugger
        else:
            debugger = lambda: True

        # check that solution exists
        if not self.solution.is_primal_feasible():
            warnings.warn('no solution exists!')
            debugger()
            return True

        # set pointers to thinks
        sol = self.solution
        data = self.data
        names = self.names
        info = self.info

        # agreement
        objval = sol.get_objective_value()
        total_agreement = np.array(sol.get_values(names['total_agreement']))
        agree_pos = np.array(sol.get_values(names['agree_pos']))
        agree_neg = np.array(sol.get_values(names['agree_neg']))

        # discrepancy
        total_error_gap = np.array(sol.get_values(names['total_error_gap']))
        max_discrepancy = int(self.vars.get_upper_bounds(names['total_error_gap']))
        min_discrepancy = int(self.vars.get_lower_bounds(names['total_error_gap']))

        # coefficients
        theta_pos = np.array(sol.get_values(names['theta_pos']))
        theta_neg = np.array(sol.get_values(names['theta_neg']))
        theta_sign = np.array(sol.get_values(names['theta_sign']))
        theta = theta_pos + theta_neg

        # other key information
        margin = np.abs(info['margin'])
        n_samples = data['n_counts_pos'].sum() + data['n_counts_neg'].sum()
        baseline_ub = self.baseline_stats['upperbound']
        baseline_lb = self.baseline_stats['lowerbound']
        epsilon = max_discrepancy - baseline_ub

        # total agreement between predictions of baseline classifier and trained classifier
        try:
            assert np.isclose(objval, total_agreement)
            expected_agreement = data['n_counts_pos'].dot(agree_pos) + data['n_counts_neg'].dot(agree_neg)
            assert np.isclose(total_agreement, expected_agreement)
        except AssertionError as e:
            print('check total agreement', e)
            debugger()

        # compute agreement for positive predictions / negative predictions
        S_pos = data['U_pos'].dot(theta)
        S_neg = data['U_neg'].dot(theta)
        expected_agree_pos = np.greater_equal(S_pos, margin)
        expected_agree_neg = np.less_equal(S_neg, -margin)

        # throw error if z[i] == 0 but s[i] >= margin
        bug_idx = np.flatnonzero((expected_agree_pos == 1) & (agree_pos == 0))
        bug_idx = np.setdiff1d(bug_idx, np.flatnonzero(np.isclose(S_pos, margin)))
        try:
            assert len(bug_idx) == 0
        except AssertionError as e:
            print('BUG: score[i] <  margin but agree_pos[i] = 0 for %1.0f points' % len(bug_idx))
            debugger()

        # throw error if z[i] == 0 but s[i] < margin
        bug_idx = np.flatnonzero((expected_agree_neg == 1) & (agree_neg == 0))
        bug_idx = np.setdiff1d(bug_idx, np.flatnonzero(np.isclose(S_neg, -margin)))
        try:
            assert len(bug_idx) == 0
        except AssertionError as e:
            print('BUG: score[i] >  -margin but agree_neg[i] = 0 for %1.0f points' % len(bug_idx))
            debugger()

        # check actual agreement
        yhat = np.sign(self.original_data['X'].dot(theta))
        assert np.array_equal(data['B'], np.sign(self.original_data['X'].dot(self.baseline_coefs)))
        actual_agreement = np.equal(yhat, data['B']).sum()
        if not np.isclose(total_agreement, actual_agreement):
            warnings.warn('total_agreement != true_agreement\n%1.0f != %1.0f' % (total_agreement, actual_agreement))

        # check total discrepancy
        try:
            expected_disc_pos = np.multiply(np.logical_not(agree_pos), data['n_counts_pos'])
            expected_disc_neg = np.multiply(np.logical_not(agree_neg), data['n_counts_neg'])
            expected_discrepancy = data['LU_pos'].dot(expected_disc_pos) + data['LU_neg'].dot(expected_disc_neg)
            assert np.isclose(total_error_gap, expected_discrepancy)
            assert min_discrepancy <= total_error_gap <= max_discrepancy
        except AssertionError as e:
            print('check total discrepancy', e)
            debugger()

        # issue warning if the discrepancy is too high
        if total_error_gap > (n_samples // 2):
            warnings.warn('trained classifier disagrees with baseline on more than 50% of samples%\ntotal_error_gap = %1.0f\nn_samples = %1.0f' % (total_agreement, n_samples))

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
            self._print_flag = DiscrepancyMIP.PRINT_FLAG
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
            self._parallel_flag = DiscrepancyMIP.PARALLEL_FLAG
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


    #### helper functions

    def bound_error_gap(self, lb = None, ub = None):
        """
        :param lb:
        :param ub:
        :param strategy:
        :return:
        """
        assert ub is None or isinstance(ub, (float, int))
        assert lb is None or isinstance(lb, (float, int))

        # preconditions
        if ub is not None:
            self.vars.set_upper_bounds(self.names['total_error_gap'], float(ub))

        if lb is not None:
            self.vars.set_lower_bounds(self.names['total_error_gap'], float(lb))


    def add_mistake_cut(self, z = None):
        """
        :param z: mistake parameter
        :return:
        """

        mistake_names = self.names['agree_pos'] + self.names['agree_neg']
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


    def add_initial_solution_with_coefficients(self, coefs = None):
        """
        adds initial solutions to MIP
        :param coefs:
        :return:
        """
        if coefs is None:
            coefs = np.array(self.baseline_coefs)
        else:
            coefs = np.array(coefs).flatten()
            assert len(coefs) == len(self.baseline_coefs)
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

