from eqm.paths import *
from eqm.classifier_helper import get_classifier_from_coefficients
from scripts.flipped_training import *
import numpy as np
from eqm.debug import ipsh


######### dashboard ##########

data_names = [
    'compas_arrest', 'compas_violent',
    'compas_arrest_small', 'compas_violent_small',
    'pretrial_CA_arrest', 'pretrial_CA_fta',
    'pretrial_NY_arrest', 'pretrial_NY_fta',
    'recidivism_CA_arrest', 'recidivism_CA_drug',
    'recidivism_NY_arrest', 'recidivism_NY_drug',
]
data_names = ['recidivism_NY_drug']

fold_id = "K05N01"

##############################
# function is run at bottom of file


def is_legal_disc_path_row(row, df):
    epsilon_is_greater = df.query('epsilon >= %s' % row['epsilon'])
    if epsilon_is_greater.shape[0]:
        upper_bound = epsilon_is_greater['train_discrepancy'].min()
        return row['train_discrepancy'] <= upper_bound
    else:
        return True


def results_to_plot_csv(info):
    """
    Calculates the discrepancy vs. the baseline for each model
    then appends to the results dataframes and saves to csv.
    :param info: a dictionary with the data_name and fold_id
    :return: None. Saves results to disk instead.
    """

    assert 'data_name' in info and 'fold_id' in info

    # get the results filenames to read
    processed_file = results_dir / data_name / get_processed_file_name(info)
    discrepancy_file = results_dir / data_name / get_discrepancy_file_name(info)

    print("Loading baseline and flipped results from: %s" % processed_file)
    print("Loading discrepancy results from: %s" % discrepancy_file)

    # get the output filenames
    processed_output = processed_file.with_suffix(".csv")
    discrepancy_output = discrepancy_file.with_suffix(".csv")

    # checks
    assert processed_file.exists(), "File does not exist: %s" % processed_file
    assert discrepancy_file.exists(), "File does not exist: %s" % discrepancy_file

    # read the results from disk
    with open(processed_file, "rb") as f:
        processed = dill.load(f)
    with open(discrepancy_file, "rb") as f:
        discrepancy = dill.load(f)

    # read the data
    data_file = data_dir / (data_name + "_processed.pickle")
    with open(data_file, "rb") as f:
        data_bundle = dill.load(f)

    # separate train and test
    data = filter_data_to_fold(data = data_bundle['data'],
                               cvindices = data_bundle['cvindices'],
                               fold_id = processed['info']['fold_id'],
                               fold_num = processed['info']['fold_num'],
                               include_validation = True)

    # unpack
    X, Y = data['X'], data['Y']
    X_test, Y_test = data['X_validation'], data['Y_validation']

    # extract the results dataframes
    processed_df = processed['results_df']
    discrepancy_df = discrepancy['results_df']

    # get the classifiers
    p_classifiers = processed_df['coefficients'].apply(get_classifier_from_coefficients)
    disc_classifiers = discrepancy_df['coefficients'].apply(get_classifier_from_coefficients)

    # compute predictions of each model
    train_predictions_p = np.array([clf.predict(X) for clf in p_classifiers])
    test_predictions_p = np.array([clf.predict(X_test) for clf in p_classifiers])
    train_predictions_disc = np.array([clf.predict(X) for clf in disc_classifiers])
    test_predictions_disc = np.array([clf.predict(X_test) for clf in disc_classifiers])

    # get the baseline model predictions
    baseline_idx = np.array(np.where(processed_df.model_type == "baseline")).flatten()
    assert len(baseline_idx) == 1
    baseline_train_predictions = train_predictions_p[baseline_idx]
    baseline_test_predictions = test_predictions_p[baseline_idx]

    # compute error rate of each model
    baseline_test_error = np.mean(baseline_test_predictions != Y_test)
    test_error_p = np.mean(test_predictions_p != Y_test, axis=1)
    test_error_disc = np.mean(test_predictions_disc != Y_test, axis=1)
    baseline_train_error = np.mean(baseline_train_predictions != Y)
    train_error_p = np.mean(train_predictions_p != Y, axis=1)
    train_error_disc = np.mean(train_predictions_disc != Y, axis=1)

    # compute test error gap of each model
    test_error_gap_p = test_error_p - baseline_test_error
    test_error_gap_disc = test_error_disc - baseline_test_error

    # get the vector of discrepancies versus the baseline model (for processed)
    train_discrepancies_p = np.mean(train_predictions_p != baseline_train_predictions, axis=1)
    test_discrepancies_p = np.mean(test_predictions_p != baseline_test_predictions, axis=1)
    # scaled_test_discrepancies_p = test_discrepancies_p * X.shape[0] / X_test.shape[0]

    # get the vector of discrepancies versus the baseline model (for discrepancy)
    train_discrepancies_disc = np.mean(train_predictions_disc != baseline_train_predictions, axis=1)
    test_discrepancies_disc = np.mean(test_predictions_disc != baseline_test_predictions, axis=1)
    # scaled_test_discrepancies_disc = test_discrepancies_disc * X.shape[0] / X_test.shape[0]

    # update the dataframes
    processed_df['train_error'] = train_error_p
    processed_df['test_error'] = test_error_p
    processed_df['train_discrepancy'] = train_discrepancies_p
    processed_df['test_discrepancy'] = test_discrepancies_p
    # processed_df['test_error_gap'] = test_error_gap_p * X.shape[0]
    # processed_df['validation_error_scaled'] = processed_df['validation_error'] * X.shape[0]

    # update the dataframes
    discrepancy_df['train_error'] = train_error_disc
    discrepancy_df['test_error'] = test_error_disc
    discrepancy_df['train_discrepancy'] = train_discrepancies_disc
    discrepancy_df['test_discrepancy'] = test_discrepancies_disc
    # discrepancy_df['test_error_gap'] = test_error_gap_disc * X.shape[0]

    # clean the discrepancy path
    is_legal = discrepancy_df.apply(is_legal_disc_path_row, axis=1, df=discrepancy_df)
    discrepancy_df = discrepancy_df[is_legal]

    # save the dataframes to disk as csv files
    processed_df.to_csv(processed_output)
    discrepancy_df.to_csv(discrepancy_output)

    print("Saved processed results to: %s" % processed_output)
    print("Saved discrepancy results to: %s" % discrepancy_output)


####################


# saves the prepped csv for each dataset specified above
for data_name in data_names:
    print("-" * 10, data_name, "-" * 10)
    info = {'fold_id': fold_id, 'data_name': data_name}
    results_to_plot_csv(info)

