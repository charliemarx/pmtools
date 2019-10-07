
from eqm.cross_validation import filter_data_to_fold
from eqm.paths import *
from scripts.flipped_training import *
import dill
from scripts.reporting import *
from eqm.classifier_helper import get_classifier_from_coefficients
from eqm.debug import ipsh
from eqm.data import remove_variable

#############################################
data_name = "compas_arrest_small"
fold_id = "K05N01"

# settings just for disparity checks
epsilon = 0.01

#############################################


# given a results df row with a classifier, and a feature matrix, returns the vector of predictions
def df_predict_handle(row, X):
    return row['clf'].predict(x)[0]


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


# def subgroup_disparity_analysis(info, subgroup_getter, protected_subgroup, epsilon=0.01):

# handle the case where compas_small datasets have balanced csvs by loading un-_small data instead
if "compas" in data_name:
    altered_data_name = data_name.replace('_small', '')
else:
    altered_data_name = data_name

info = {'data_name': data_name,
        'fold_id': fold_id}
altered_info = {'data_name': altered_data_name,
        'fold_id': fold_id}

# set up directories
output_dir = paper_dir / info['data_name']
processed_file = output_dir / get_processed_file_name(info)
discrepancy_file = output_dir / get_discrepancy_file_name(info)
data_file = data_dir / (altered_info['data_name'] + "_processed.pickle")

# get the models
with open(processed_file, "rb") as f:
    processed = dill.load(f)
with open(discrepancy_file, "rb") as f:
    discrepancy = dill.load(f)

# get the data
with open(data_file, "rb") as f:
    data = dill.load(f)
    data = filter_data_to_fold(data['data'], data['cvindices'], info['fold_id'], fold_num=1, include_validation=True)

# get the unbalanced data
unbalanced = load_data_from_csv(data_file.with_suffix(".csv"))

#
if "compas" in data_name and "_small" in data_name:
    for feat in ['race_is_causasian',
                 'race_is_african_american',
                 'race_is_hispanic',
                 'race_is_other']:
        if feat in data['variable_names']:
            data = remove_variable(data, feat)
        if feat in unbalanced['variable_names']:
            unbalanced = remove_variable(unbalanced, feat)

# extract the balanced data
X, Y = data['X'], data['Y']
X_test, Y_test = data['X_validation'], data['Y_validation']

# extract the unbalanced data
UX, UY = unbalanced['X'], unbalanced['Y']


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

proc_metrics = pd.DataFrame.from_records(proc_results['clf'].apply(get_metrics, X=X, Y=Y, X_test=X_test, Y_test=Y_test, metrics=["train_error", "test_error"]))
proc_results = pd.concat([proc_results, proc_metrics], axis=1)

disc_metrics = pd.DataFrame.from_records(disc_results['clf'].apply(get_metrics, X=X, Y=Y, X_test=X_test, Y_test=Y_test, metrics=["train_error", "test_error"]))
disc_results = pd.concat([disc_results, disc_metrics], axis=1)

proc_level_set = proc_results.query('train_error <= %s' % (baseline_train_error + epsilon))
disc_level_set = disc_results.query('train_error <= %s' % (baseline_train_error + epsilon))
level_set_clfs = proc_level_set['clf'].append(disc_level_set['clf'])


def get_multiplicity(baseline_clf, level_set_clfs, X):
    baseline_preds = baseline_clf.predict(X)
    num_instances = X.shape[0]

    ambiguous_set = set()
    num_max_discrepancy = 0
    for clf in level_set_clfs:
        preds = clf.predict(X)
        conflicts = np.not_equal(preds, baseline_preds)
        ambiguous_set.update(np.where(conflicts)[0])
        discrepancy = np.sum(conflicts)
        num_max_discrepancy = max(discrepancy, num_max_discrepancy)

    num_ambiguous = len(ambiguous_set)
    ambiguity = num_ambiguous / num_instances
    max_discrepancy = num_max_discrepancy / num_instances
    results = {"n": X.shape[0],
            "ambiguity": ambiguity,
            "num_ambiguous": num_ambiguous,
            "max_discrepancy": max_discrepancy,
            "num_max_discrepancy": num_max_discrepancy}

    return {k: np.round(v, 3) for k, v in results.items()}


get_multiplicity(baseline_clf, level_set_clfs, X)
get_multiplicity(baseline_clf, level_set_clfs, UX)



