
from scripts.subgroup_analysis import get_compas_subgroups
from eqm.paths import *
from eqm.cross_validation import filter_data_to_fold
from eqm.classifier_helper import get_classifier_from_coefficients
from scripts.flipped_training import get_processed_file_name, get_discrepancy_file_name
import dill
import numpy as np
import pandas as pd


def get_metrics(clf, X, Y, X_test, Y_test):
    train_preds = clf.predict(X)
    test_preds = clf.predict(X_test)

    train_error = np.mean(train_preds != Y)
    test_error = np.mean(test_preds != Y_test)
    train_TPR = np.mean((train_preds == Y)[Y == 1])
    test_TPR = np.mean((test_preds == Y_test)[Y_test == 1])
    train_FPR = np.mean((train_preds != Y)[Y == -1])
    test_FPR = np.mean((test_preds != Y_test)[Y_test == -1])

    metrics = {"Train Error": train_error,
               "Test Error": test_error,
               "Train TPR": train_TPR,
               "Test TPR": test_TPR,
               "Train FPR": train_FPR,
               "Test FPR": test_FPR,
               }

    return metrics

all_data_names = [
    'compas_arrest_small', 'compas_violent_small',
    'pretrial_CA_arrest', 'pretrial_CA_fta',
    'pretrial_NY_arrest', 'pretrial_NY_fta',
    'recidivism_CA_arrest', 'recidivism_CA_drug',
    'recidivism_NY_arrest', 'recidivism_NY_drug',
]
fold_id = "K05N01"
output_dir = paper_dir
table_file = output_dir / 'baseline_summary.tex'



#for data_name in all_data_names:
info = {'data_name': "compas_arrest_small", "fold_id": "K05N01"}

def get_summary_row(info):
    # set up directories
    working_dir = output_dir / info['data_name']
    processed_file = working_dir / get_processed_file_name(info)
    discrepancy_file = working_dir / get_discrepancy_file_name(info)
    data_file = data_dir / (info['data_name'] + "_processed.pickle")

    # get the baseline and flipped models
    with open(processed_file, "rb") as f:
        processed = dill.load(f)['results_df']

    # get the discrepancy models
    with open(discrepancy_file, "rb") as f:
        discrepancy = dill.load(f)['results_df']

    # load data
    with open(data_file, "rb") as f:
        data = dill.load(f)
        data = filter_data_to_fold(data['data'], data['cvindices'], fold_id=info['fold_id'], fold_num=1,
                                   include_validation=True)

    X, Y = data['X'], data['Y']
    X_test, Y_test = data['X_validation'], data['Y_validation']

    # make a classifier for each model in the df
    processed['clf'] = processed['coefficients'].apply(get_classifier_from_coefficients)
    discrepancy['clf'] = discrepancy['coefficients'].apply(get_classifier_from_coefficients)

    # get the baseline classifier
    baseline = processed.query("model_type == 'baseline'").iloc[0]
    baseline_clf = baseline['clf']
    baseline_metrics = get_metrics(baseline_clf, X, Y, X_test, Y_test)

    baseline_metrics.update({"Dataset": info['data_name']})

    return baseline_metrics


summary_table = pd.DataFrame([get_summary_row({'data_name': data_name, "fold_id": fold_id}) for data_name in all_data_names])
summary_table['Dataset'] = summary_table['Dataset'].apply(lambda x: "\\texttt{%s}" % x.replace("_small", "").replace("_", "\\_"))
metric_cols = ["Test Error", "Test FPR", "Test TPR", "Train Error", "Train FPR", "Train TPR"]
for col in metric_cols:
    summary_table[col] *= 100
    summary_table[col] = summary_table[col].apply(lambda x: "%1.1f" % x)

summary_table = summary_table[["Dataset",
                               "Train Error",
                               "Train FPR",
                               "Train TPR",
                               "Test Error",
                               "Test FPR",
                               "Test TPR",
                               ]]

summary_table.to_latex(buf = table_file,
                       escape = False,
                       index = False,
                       header = True)

