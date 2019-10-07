from eqm.cross_validation import filter_data_to_fold
from eqm.paths import *
from scripts.flipped_training import *
import dill
from scripts.reporting import *
from eqm.classifier_helper import get_classifier_from_coefficients
from eqm.debug import ipsh

#
# # get the race of each individual in the compas dataset
# def get_compas_subgroups(data):
#     race_is_caucasian = data['X'][:, data['variable_names'].index('race_is_causasian')]
#     race_is_african_american = data['X'][:, data['variable_names'].index('race_is_african_american')]
#     race_is_hispanic = data['X'][:, data['variable_names'].index('race_is_hispanic')]
#     race_is_other = data['X'][:, data['variable_names'].index('race_is_other')]
#     subgroups = [
#         'caucasian' if race_is_caucasian[i] else
#         'african_american' if race_is_african_american[i] else
#         'hispanic' if race_is_hispanic[i] else
#         'other' for i in range(len(data['X']))
#         ]
#
#     return np.array(subgroups)


def get_compas_subgroups(outcome="arrest"):
    compas_file = data_dir / ("compas_%s_processed.pickle" % outcome)
    with open(compas_file, "rb") as f:
        data = dill.load(f)
        data = filter_data_to_fold(data['data'], data['cvindices'], info['fold_id'], fold_num=1)
        race_is_caucasian = data['X'][:, data['variable_names'].index('race_is_causasian')]
        race_is_african_american = data['X'][:, data['variable_names'].index('race_is_african_american')]
        race_is_hispanic = data['X'][:, data['variable_names'].index('race_is_hispanic')]
        race_is_other = data['X'][:, data['variable_names'].index('race_is_other')]
        subgroups = [
            'caucasian' if race_is_caucasian[i] else
            'african_american' if race_is_african_american[i] else
            'hispanic' if race_is_hispanic[i] else
            'other' for i in range(len(data['X']))
            ]
        return np.array(subgroups)

def get_compas_subgroup_getter(outcome):
    return lambda : get_compas_subgroups(outcome=outcome)

#############################################
data_name = "compas_arrest_small"
subgroup_getter = get_compas_subgroup_getter(outcome="arrest")
fold_id = "K05N01"

# settings just for disparity checks
protected_subgroup = "african_american"
epsilon = 0.01

#############################################

info = {'data_name': data_name,
        'fold_id': fold_id}


# given a results df row and an instance, return the prediction
def df_predict_handle(row, x):
    return row['clf'].predict(x)[0]

# returns as "%1.1e" if abs(num) < 0.1 else returns as `format`
def scientific_if_small(num, format="%1.2f"):
    if abs(num) < 0.01 and num != 0:
        return "%1.1e" % abs(num)
    else:
        return format % abs(num)

def num_to_signed(num, format="%1.2f"):
    sign = '+' if num >= 0 else '-'
    return ("%s" + format) % (sign, abs(num))

def subgroup_ambiguity_analysis(info, subgroup_getter):

    # set up directories
    output_dir = paper_dir / info['data_name']
    processed_file = output_dir / get_processed_file_name(info)
    data_file = data_dir / (info['data_name'] + "_processed.pickle")

    # get the models
    with open(processed_file, "rb") as f:
        processed = dill.load(f)

    # get the data
    with open(data_file, "rb") as f:
        data = dill.load(f)
        data = filter_data_to_fold(data['data'], data['cvindices'], info['fold_id'], fold_num=1)

    # make a classifier for each model in the df
    results = processed['results_df']
    results['clf'] = results['coefficients'].apply(get_classifier_from_coefficients)

    # get the baseline classifier
    assert results.query("model_type == 'baseline'").shape[0] == 1
    baseline = results.query("model_type == 'baseline'").iloc[0]
    baseline_clf = baseline['clf']
    baseline_train_error = baseline['train_error']

    # compress data and get subgroups vector for analysis (e.g. race)
    U, x_to_u_idxs, counts = np.unique(data['X'], axis=0, return_counts=True, return_index=True)
    subgroups = subgroup_getter()
    subgroups = subgroups[x_to_u_idxs]

    # get the train error of the best flipped model for each instance
    flipped_train_errors = []
    for i, instance in enumerate(U):
        instance = np.expand_dims(instance, 0)
        baseline_pred = baseline_clf.predict(instance)[0]
        preds = results.apply(df_predict_handle, axis=1, x=instance)

        min_flipped_train_error = results[preds != baseline_pred]['train_error'].min()
        flipped_train_errors.append(min_flipped_train_error)

    # subgroup analysis dataframe
    df = pd.DataFrame({
        'subgroup': subgroups,
        'flipped_error': flipped_train_errors,
        'num_instances': counts,
        })
    # get number of instances per subgroup
    group_counts = {grp: df.query("subgroup == '%s'" % grp)['num_instances'].sum() for grp in np.unique(df['subgroup'])}

    # check the percent of epsilon-flippable instances per subgroup
    epsilon = 0.01
    subgroup_multiplicity = {}
    for sgroup in np.unique(df['subgroup']):
        grp = df.query("subgroup == '%s'" % sgroup)
        flippable = grp['flipped_error'] <= baseline_train_error + epsilon
        num_flippable = grp[flippable]['num_instances'].sum()
        pct_flippable = np.round(num_flippable / group_counts[sgroup], 4)

        subgroup_multiplicity[sgroup] = {'num_flippable': num_flippable,
                           'pct_flippable': pct_flippable,
                           'num_instances': group_counts[sgroup],
                           }
    return subgroup_multiplicity


def get_metrics(clf, X, Y, X_test, Y_test, metrics=None):
    train_preds = clf.predict(X)
    test_preds = clf.predict(X_test)

    train_error = np.mean(train_preds != Y)
    test_error = np.mean(test_preds != Y_test)
    train_TPR = np.mean((train_preds == Y)[Y == 1])
    test_TPR = np.mean((test_preds == Y_test)[Y_test == 1])
    train_FPR = np.mean((train_preds != Y)[Y == -1])
    test_FPR = np.mean((test_preds != Y_test)[Y_test == -1])

    results = {"train_error": train_error,
               "test_error": test_error}

    assert all(k in results for k in metrics)
    if metrics is not None:
        results = {k: results[k] for k in metrics}

    return results


def get_disparity(clf, X, subgroups, protected_name):
    is_protected = np.array([g == protected_name for g in subgroups])
    X_protected = X[is_protected]
    X_unprotected = X[~is_protected]
    protected_pos_rate = np.mean(clf.predict(X_protected) == 1.0)
    unprotected_post_rate = np.mean(clf.predict(X_unprotected) == 1.0)

    disparity = protected_pos_rate - unprotected_post_rate
    return disparity


def subgroup_disparity_analysis(info, subgroup_getter, protected_subgroup, epsilon=0.01):

    # set up directories
    output_dir = paper_dir / info['data_name']
    processed_file = output_dir / get_processed_file_name(info)
    discrepancy_file = output_dir / get_discrepancy_file_name(info)
    data_file = data_dir / (info['data_name'] + "_processed.pickle")

    # get the models
    with open(processed_file, "rb") as f:
        processed = dill.load(f)
    with open(discrepancy_file, "rb") as f:
        discrepancy = dill.load(f)

    # get the data
    with open(data_file, "rb") as f:
        data = dill.load(f)
        data = filter_data_to_fold(data['data'], data['cvindices'], info['fold_id'], fold_num=1, include_validation=True)

    X, Y = data['X'], data['Y']
    X_test, Y_test = data['X_validation'], data['Y_validation']
    subgroups = subgroup_getter()
    # subgroups = [g if g == "african_american" else "other" for g in subgroups]

    # make a classifier for each model in the df
    proc_results = processed['results_df']
    disc_results = discrepancy['results_df']

    proc_results['clf'] = proc_results['coefficients'].apply(get_classifier_from_coefficients)
    disc_results['clf'] = disc_results['coefficients'].apply(get_classifier_from_coefficients)

    # get the baseline classifier
    assert proc_results.query("model_type == 'baseline'").shape[0] == 1
    baseline = proc_results.query("model_type == 'baseline'").iloc[0]
    baseline_clf = baseline['clf']
    baseline_train_error = baseline['train_error']
    baseline_disparity = get_disparity(baseline_clf, X=X, subgroups=subgroups, protected_name=protected_subgroup)

    proc_metrics = pd.DataFrame.from_records(proc_results['clf'].apply(get_metrics, X=X, Y=Y, X_test=X_test, Y_test=Y_test, metrics=["train_error", "test_error"]))
    proc_results = pd.concat([proc_results, proc_metrics], axis=1)

    disc_metrics = pd.DataFrame.from_records(disc_results['clf'].apply(get_metrics, X=X, Y=Y, X_test=X_test, Y_test=Y_test, metrics=["train_error", "test_error"]))
    disc_results = pd.concat([disc_results, disc_metrics], axis=1)

    proc_level_set = proc_results.query('train_error <= %s' % (baseline_train_error + epsilon))
    disc_level_set = disc_results.query('train_error <= %s' % (baseline_train_error + epsilon))
    level_set_clfs = proc_level_set['clf'].append(disc_level_set['clf'])

    all_metrics = pd.concat([proc_metrics, disc_metrics], axis=0)
    all_metrics_no_baseline = all_metrics.query("train_error != %s" % all_metrics['train_error'].min())
    train_error_of_best_test = all_metrics_no_baseline['train_error'][all_metrics_no_baseline['test_error'].idxmin()]
    empirical_epsilon = (train_error_of_best_test - baseline_train_error).to_list()[0]

    disparities = level_set_clfs.apply(get_disparity, X=X, subgroups=subgroups, protected_name=protected_subgroup)
    smallest_disparity = min(abs(disparities)) * np.sign(min(disparities))

    results = {"baseline_disparity": baseline_disparity,
               "best_disparity": smallest_disparity,
               "difference": baseline_disparity - smallest_disparity,
               "epsilon": epsilon,
               "empirical_epsilon": empirical_epsilon,
               "level_set_size": len(disparities),
               "error_&_disparity_level_set_size": sum(abs(disparities - baseline_disparity) <= epsilon),
               }

    results = {k: num_to_signed(v) for k, v in results.items()}
    # ipsh()
    return results

# subgroup_ambiguity_analysis(info=info, subgroup_getter=subgroup_getter)
subgroup_disparity_analysis(info=info, subgroup_getter=subgroup_getter, protected_subgroup=protected_subgroup, epsilon=epsilon)
