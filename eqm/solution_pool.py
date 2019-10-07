import numpy as np
import pandas as pd
import warnings
from eqm.mip import ZeroOneLossMIP

from eqm.debug import ipsh

class SolutionPool(object):
    """
    helper class used to create/manipulate a queue of solutions and objective values
    """

    names = ['solution', 'coefficients', 'objval', 'lowerbound', 'prediction_constraint']

    def __init__(self, mip = None, df = None):

        if df is None:
            self._df = pd.DataFrame(columns = self.names)
        else:
            assert isinstance(df, pd.DataFrame)
            self._df = df.copy()[self.names]

        if mip is not None:
            self.add_from_mip(mip)



    def includes(self, solution):
        """
        :param solution: solution vector
        :return: True if there exists another solution in this object that matches th
        """
        return any(solution == old for old in self._df['solution'])


    def add_from_mip(self, mip, prediction_constraint = None, add_full_solution_pool = True):
        """
        :param mip:
        :param prediction_constraint:
        :param add_full_solution_pool:
        :return:
        """
        assert isinstance(mip, ZeroOneLossMIP)

        if not mip.solution.is_primal_feasible():
            warnings.warn('no solution exists!')
            return

        s = mip.solution
        lb = s.MIP.get_best_objective()

        # record current solution
        raw_df = {'objval': [s.MIP.get_cutoff()], 'solution': [s.get_values()]}

        # add full mip solution pool to the pool
        if add_full_solution_pool:
            pool = s.pool
            for k in range(1, pool.get_num()):
                # note that we start from 1 since the first solution in the pool is usually the optimal solution
                solution, objval = pool.get_values(k), pool.get_objective_value(k)
                if not any(solution == sol for sol in raw_df['solution']):
                    raw_df['objval'].append(objval)
                    raw_df['solution'].append(solution)

        assert len(set(map(len, raw_df.values()))) == 1,  'all lists in dict should be of the same length'
        df = [{'solution': sol, 'objval': val} for sol, val in zip(raw_df['solution'], raw_df['objval']) if not self.includes(sol)]
        if len(df) > 0:
            df = pd.DataFrame(df)
            df['coefficients'] = [mip.solution_to_coefs(w) for w in df['solution']]
            df['lowerbound'] = lb
            df['prediction_constraint'] = [prediction_constraint] * len(df)
            self._df = self._df.append(df, sort = False).reset_index(drop = True)


    def add(self, solution, coefficients, objval, lowerbound, prediction_constraint = None):
        """
        :param solution:
        :param coefficients:
        :param objval:
        :param lowerbound:
        :param prediction_constraint:
        :return:
        """

        if isinstance(objval, (list, np.ndarray)):
            # convert placeholder for prediction constraints to list of appropriate size
            if prediction_constraint is None:
                prediction_constraint = [None] * len(solution)

            assert all(len(param) == len(objval) for param in (solution, coefficients, lowerbound, prediction_constraint))
            param_dict = {'solution': solution,
                          'objval': objval,
                          'coefficients': coefficients,
                          'lowerbound': lowerbound,
                          'prediction_constraint': prediction_constraint}
        else:

            param_dict = {'solution': [solution],
                          'objval': [objval],
                          'coefficients': [coefficients],
                          'lowerbound': [lowerbound],
                          'prediction_constraint': [prediction_constraint]}

        new_df = pd.DataFrame.from_dict(param_dict)
        self._df = self._df.append(new_df, sort = False).reset_index(drop = True)


    def get_solutions_with_pred(self, x, yhat):
        """
        Returns a new solution pool containing only those solutions which satisfy h(x)=yhat.
        The current solution pool is left unchanged.
        :param x:
        :param yhat:
        :return:
        """
        preds = self.get_preds(x)
        include = self._df[preds == yhat]
        sol_pool = SolutionPool()
        sol_pool.add(solution=include['solution'].to_list(),
                     coefficients=include['coefficients'].to_list(),
                     objval=include['objval'].to_list(),
                     lowerbound=include['lowerbound'].to_list(),
                     prediction_constraint=include['prediction_constraint'].to_list())
        return sol_pool


    def get_preds(self, x):
        """
        :param x: The instance on which to compute predictions.
        :return: The predictions of each solution.
        """
        W = np.array(self.coefficients)
        preds = np.sign(np.dot(W, x))
        return preds


    def get_best_solution(self):
        best_idx = self._df['objval'].idxmin()
        best_solution = self._df.iloc[best_idx].to_dict()
        return best_solution


    def merge(self, pool):
        self._df.append(pool, sort=False).reset_index(drop=True)


    def deduplicate(self):
        raise NotImplementedError()
        # todo: Implement this by converting solution to tuple so it is hashable,
        # then deduplicating on the tuple column


    def get_df(self):
        return self._df.copy(deep=True)


    def clear(self):
        self._df.drop(self._df.index, inplace=True)


    @property
    def size(self):
        return self._df.shape[0]


    @property
    def objvals(self):
        return self._df['objval'].tolist()


    @property
    def solutions(self):
        return self._df['solution'].tolist()


    @property
    def coefficients(self):
        return self._df['coefficients'].tolist()


    @property
    def lowerbounds(self):
        return self._df['lowerbound'].tolist()


    @property
    def prediction_constraints(self):
        return self._df['prediction_constraint']


    def __len__(self):
        return len(self._df)


    def __repr__(self):
        return self._df.__repr__()


    def __str__(self):
        return self._df.__str__()
