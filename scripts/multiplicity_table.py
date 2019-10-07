from eqm.cross_validation import filter_data_to_fold
from eqm.paths import *
from scripts.flipped_training import get_processed_file_name, get_discrepancy_file_name
import dill
from scripts.reporting import *
from eqm.classifier_helper import get_classifier_from_coefficients

np.set_printoptions(precision=4)

#############################################
output_dir = paper_dir # results_dir or paper_dir
data_name = "compas_violent"
fold_id = "K05N01"

# todo: format the comparison table
#   - change error rates to %ss (33.1 \%)

# todo: add abiilty to produce this for GLMNET files

#############################################

info = {
    'data_name': data_name,
    'fold_id': fold_id,
    'metric_name': '01', #
    }

info['data_file'] = data_dir / ('%s_processed.pickle' % info['data_name'])
info['table_file'] = output_dir / ('%s_%s_comparison_table.tex' % (info['data_name'], info['metric_name']))
info['model_file_header'] = '%s_%s_model' % (info['data_name'], info['metric_name'])

data, _ = load_processed_data(info['data_file'])

# given a results df row and an instance, return the prediction
def df_predict_handle(row, x):
    return row['clf'].predict(x)[0]


def multiplicity_table(info):

    # set up directories
    working_dir = output_dir / info['data_name']
    processed_file = working_dir / get_processed_file_name(info)
    discrepancy_file = working_dir / get_discrepancy_file_name(info)

    # get the baseline and flipped models
    with open(processed_file, "rb") as f:
        processed = dill.load(f)['results_df']

    # get the discrepancy models
    with open(discrepancy_file, "rb") as f:
        discrepancy = dill.load(f)['results_df']

    # load data
    with open(info['data_file'], "rb") as f:
        data = dill.load(f)
        data = filter_data_to_fold(data['data'], data['cvindices'], fold_id=info['fold_id'], fold_num=1, include_validation=True)

    X, Y = data['X'], data['Y']
    X_test, Y_test = data['X_validation'], data['Y_validation']

    # make a classifier for each model in the df
    processed['clf'] = processed['coefficients'].apply(get_classifier_from_coefficients)
    discrepancy['clf'] = discrepancy['coefficients'].apply(get_classifier_from_coefficients)

    # get the baseline classifier
    baseline = processed.query("model_type == 'baseline'").iloc[0]
    baseline_clf = baseline['clf']
    baseline_scores = baseline_clf.score(X)
    baseline_train_preds = baseline_clf.predict(X)
    baseline_test_preds = baseline_clf.predict(X_test)
    baseline_train_error = np.sum(baseline_train_preds != Y)
    baseline_train_error_rate = np.mean(baseline_train_preds != Y)
    baseline_test_error = np.sum(baseline_test_preds != Y_test)
    baseline_test_error_rate = np.mean(baseline_test_preds != Y_test)

    # get the equivalent classifier
    epsilon = discrepancy.query("epsilon <= %s" % int(0.01 * X.shape[0]))['epsilon'].max()
    equivalent = discrepancy.query("epsilon == %s" % epsilon).iloc[0]
    equivalent_clf = equivalent['clf']
    equivalent_train_preds = equivalent_clf.predict(X)
    equivalent_test_preds = equivalent_clf.predict(X_test)
    equivalent_train_error = np.sum(equivalent_train_preds != Y)
    equivalent_train_error_rate = np.mean(equivalent_train_preds != Y)
    equivalent_test_error = np.sum(equivalent_test_preds != Y_test)
    equivalent_test_error_rate = np.mean(equivalent_test_preds != Y_test)
    equivalent_train_discrepancy = np.sum(equivalent_train_preds != baseline_train_preds)
    equivalent_test_discrepancy = np.sum(equivalent_test_preds != baseline_test_preds)
    equivalent_train_discrepancy_rate = np.mean(equivalent_train_preds != baseline_train_preds)
    equivalent_test_discrepancy_rate = np.mean(equivalent_test_preds != baseline_test_preds)

    is_flipped = np.logical_and(baseline_train_preds == 1, equivalent_train_preds == -1)
    is_flipped_nonzero = np.logical_and(is_flipped, abs(baseline_scores) > 0.001)


    flipped_train_errors = []
    for instance in np.unique(X[is_flipped_nonzero], axis=0):
        # get the train error of the best flipped model for each instance
        instance = np.expand_dims(instance, 0)
        baseline_pred = baseline_clf.predict(instance)[0]
        preds = processed.apply(df_predict_handle, axis=1, x=instance)
        min_flipped_train_error = processed[preds != baseline_pred]['objval'].min()
        flipped_model = processed[preds != baseline_pred].query('objval == %s' % min_flipped_train_error).iloc[0]
        flipped_train_errors.append({'min_flipped_train_error': min_flipped_train_error,
                                     'instance': instance,
                                     'model': flipped_model})

    # choose the instance to show
    flipped_info = min(flipped_train_errors, key=lambda x: x["min_flipped_train_error"])
    x_i = flipped_info['instance']

    # get the flipped classifier
    flipped = flipped_info['model']
    flipped_clf = flipped['clf']
    flipped_train_preds = flipped_clf.predict(X)
    flipped_test_preds = flipped_clf.predict(X_test)
    flipped_train_error = np.sum(flipped_train_preds != Y)
    flipped_train_error_rate = np.mean(flipped_train_preds != Y)
    flipped_test_error = np.sum(flipped_test_preds != Y_test)
    flipped_test_error_rate = np.mean(flipped_test_preds != Y_test)
    flipped_train_discrepancy = np.sum(flipped_train_preds != baseline_train_preds)
    flipped_test_discrepancy = np.sum(flipped_test_preds != baseline_test_preds)
    flipped_train_discrepancy_rate = np.mean(flipped_train_preds != baseline_train_preds)
    flipped_test_discrepancy_rate = np.mean(flipped_test_preds != baseline_test_preds)


    results_df = pd.DataFrame(columns=['model_type', 'coefficients',
                                       'train_error_rate', 'test_error_rate',
                                       'train_discrepancy_rate', 'test_discrepancy_rate',
                                       'score_xi', 'prediction_xi'])

    results_df = results_df.append({'model_type': 'baseline',
                                    'coefficients': baseline['coefficients'],
                                    'train_error_rate': baseline_train_error_rate,
                                    'test_error_rate': baseline_test_error_rate,
                                    'train_discrepancy_rate': 0,
                                    'test_discrepancy_rate': 0,
                                    'score_xi': baseline_clf.score(x_i)[0],
                                    'prediction_xi': baseline_clf.predict(x_i)[0]
                                       }, ignore_index=True)

    results_df = results_df.append({'model_type': 'discrepancy',
                                    'coefficients': equivalent['coefficients'],
                                    'train_error_rate': equivalent_train_error_rate,
                                    'test_error_rate': equivalent_test_error_rate,
                                    'train_discrepancy_rate': equivalent_train_discrepancy_rate,
                                    'test_discrepancy_rate': equivalent_test_discrepancy_rate,
                                    'score_xi': equivalent_clf.score(x_i)[0],
                                    'prediction_xi': equivalent_clf.predict(x_i)[0]
                                       }, ignore_index=True)

    results_df = results_df.append({'model_type': 'flipped',
                                    'coefficients': flipped['coefficients'],
                                    'train_error_rate': flipped_train_error_rate,
                                    'test_error_rate': flipped_test_error_rate,
                                    'train_discrepancy_rate': flipped_train_discrepancy_rate,
                                    'test_discrepancy_rate': flipped_test_discrepancy_rate,
                                    'score_xi': flipped_clf.score(x_i)[0],
                                    'prediction_xi': flipped_clf.predict(x_i)[0]
                                       }, ignore_index=True)


    x_i_features = pd.Series(x_i[0], index=data['variable_names'])
    results = {'results_df': results_df, 'x_i': x_i_features}
    return results


def num_to_signed(num, format="%1.1f"):
    sign = '+' if num >= 0 else '-'
    return ("%s" + format) % (sign, abs(num))


# returns as "%1.1e" if abs(num) < 0.1 else returns as `format`
def scientific_if_small(num, format="%1.1f"):
    if abs(num) < 0.1 and num != 0:
        return "%1.1e" % abs(num)
    else:
        return format % abs(num)


def score_function_table(coefs, data, fmt = '%1.1f', scale = 100.0, coefs_per_row = 2):
    """
    %
    converts coefficient vector of linear model into a score function that can be included in latex documents
    :param coefs: coefficient vector
    :param data: data
    :param fmt: format string for coefficients
    :param scale: positive numbere to scale coefficients in table
                  coefs in table are rescaled so that sum(abs(w[j])) == scale
    :param coefs_per_row: # of coefficients to show in each row
    :return: string with the code of a latex tabular environment (can be printed/saved to disk)
    """

    # basic assertions
    coefs = np.array(coefs).flatten()
    assert np.isfinite(coefs).all()

    assert scale > 0.0
    scale = scale / np.abs(coefs).sum()

    coefs_per_row = int(coefs_per_row)
    assert 1 <= coefs_per_row <= len(coefs)

    # print coefficients and names for variables
    var_idx = get_variable_indices(data, include_intercept = False)
    print_coefs = [scale * coefs[i] for i in var_idx]
    print_names = [data['variable_names'][i] for i in var_idx]

    # add intercept
    print_coefs.append(coefs[INTERCEPT_IDX])
    print_names.append('') # remove the name for the intercept

    assert len(print_coefs) == len(print_names)

    # reformat variable names
    entries = []
    for pw, pn in zip(print_coefs, print_names):

        # sign
        ps = '$-$' if pw < 0.0 else '$+$'

        # coef
        pw = fmt % abs(pw)

        # name
        pn = pn.replace('_geq_', '$\\geq$')
        pn = pn.replace('_gt_', '$>$')
        pn = pn.replace('_lt_', '$<$')
        pn = pn.replace('_leq_', '$\\leq$')
        pn = pn.replace('_eq_', '$=$')
        pn = pn.replace('_is_', '$=$')
        pn = pn.replace('_bt_', '$\\in$')
        pn = pn.replace('_to_', '-')
        pn = pn.replace('_', '\\_')
        if len(pn) > 0:
            pn = '\\textfn{%s}' % pn

        # score function entry
        p = (ps, pw, pn)
        entries.append(p)

    # add empty elemnents so that we can convert into a table
    n_padding = coefs_per_row - np.remainder(len(entries), coefs_per_row)
    entries = entries + [('', '', '')] * n_padding

    # convert entries into df
    df = np.reshape(entries, (-1, 3 * coefs_per_row))
    df = pd.DataFrame(df)

    # convert df into latex
    out = df.to_latex(
            column_format = '@{\;}c@{\;}r@{\,}l' * coefs_per_row,
            header = False,
            index = False,
            escape = False
            )

    # remove booktabs
    out = out.replace('\\toprule\n', '')
    out = out.replace('\\bottomrule\n', '')
    return out



table_info = multiplicity_table(info)

df = table_info['results_df']
df['metric_name'] = info['metric_name']

# save score function tables to disk
# each file will be included in latex document
cmd_list = []

for i, row in df.iterrows():

    # create table for the  model
    table = score_function_table(row['coefficients'], data)

    # save file
    save_file = '%s_%s.tex' % (info['model_file_header'], row['model_type'])
    save_file = paper_dir / save_file

    with open(save_file, 'w') as f:
        f.write(table)

    # include command to include this doc in the paper
    cmd = '\\coeftbl{%s}{%s}{%s}' % (info['data_name'], row['metric_name'], row['model_type'])
    # cmd = cmd.replace('_', '\\_')
    cmd_list.append(cmd)

df['score_function'] = cmd_list
df['prediction_xi'] = df['prediction_xi'].apply(num_to_signed, format='%d')
df['score_xi'] = (100 * df['score_xi']).apply(num_to_signed, format='%1.1f')
df['train_discrepancy_rate'] = (100 * df['train_discrepancy_rate']).apply(scientific_if_small)
df['test_discrepancy_rate'] = (100 * df['test_discrepancy_rate']).apply(scientific_if_small)
df['test_error_rate'] = (100 * df['test_error_rate']).apply(scientific_if_small)
df['train_error_rate'] = (100 * df['train_error_rate']).apply(scientific_if_small)

# reorder columns
table_df = df[['model_type',
               'score_function',
               'prediction_xi',
               'score_xi',
               'train_discrepancy_rate',
               'test_discrepancy_rate',
               'train_error_rate',
               'test_error_rate']]

# rename
table_df = table_df.rename(
        columns = {
            'score_function': 'Score Function',
            'prediction_xi': '$h(\\xb_i)$',
            'score_xi': '$score(\\xb_i)$',
            'train_discrepancy_rate': 'Discrepancy Train ',
            'test_discrepancy_rate': 'Discrepancy Test ',
            'train_error_rate': 'Error Train',
            'test_error_rate': 'Error Test',
            'model_type': "Model Type"
            }
        )

# change
table_df = table_df.transpose()

# todo: extra embellishments here
table_df.to_latex(buf = info['table_file'],
                          escape = False,
                          index = True,
                          header = False)

