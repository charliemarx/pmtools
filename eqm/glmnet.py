from rpy2.robjects import numpy2ri, r, packages
from rpy2.robjects.packages import importr, STAP
from rpy2.robjects.vectors import FloatVector
from eqm.cross_validation import *
import numpy as np
import pandas as pd
from copy import deepcopy


DEFAULT_ALPHA = 1.0
DEFAULT_NLAMBDA = 100


def fit_glm(data, family='binomial', glmnet_kwargs=None):
    """
    :param data: the data dictionary
    :param family: response type
    :param glmnet_kwargs: dictionary of keyword arguments to pass the glmnet function in R.
    :return: pandas dataframe containing the fit model parameters. Each row corresponds to a unique value for lambda.
    """
    if not packages.isinstalled(name='glmnet'):
        utils = packages.importr('utils')
        utils.chooseCRANmirror(ind=1)
        utils.install_packages('glmnet')

    if not glmnet_kwargs:
        glmnet_kwargs = {}

    # intercept should be added as a constant 1 feature, not via glmnet. Thus, always set to 'False'.
    if 'intercept' in glmnet_kwargs.keys():
        assert glmnet_kwargs['intercept'] is False, \
            "Do not add intercept in glmnet. Please add an intercept feature to the dataset instead."
    glmnet_kwargs['intercept'] = False

    # set default parameters
    if 'alpha' not in glmnet_kwargs.keys():
        glmnet_kwargs['alpha'] = DEFAULT_ALPHA
    if 'nlambda' not in glmnet_kwargs.keys():
        glmnet_kwargs['nlambda'] = DEFAULT_NLAMBDA

    # R set-up
    numpy2ri.activate()
    glmnet = importr('glmnet')

    n_row, n_col = data['X'].shape

    # transfer to R objects
    x_ = r.matrix(data['X'], nrow=n_row)
    y_ = r.matrix(data['Y'], nrow=n_row)
    weights = FloatVector(data['sample_weights'])

    output = glmnet.glmnet(x=x_, y=y_, family=family, weights=weights, **glmnet_kwargs)
    coefs = r.t(sparse_to_full_matrix(output.rx2('beta')))
    coefs = np.array(coefs)     # drop added intercept column which is fixed at 0
    lambda_ = output.rx('lambda')
    lambda_ = np.array(r.matrix(lambda_))[0]

    coef_names = data['variable_names']
    df = pd.DataFrame(coefs, columns=coef_names)

    df['lambda_'] = lambda_
    df['alpha'] = glmnet_kwargs['alpha']

    return df


def sparse_to_full_matrix(sparse_matrix):
    """
    :param sparse_matrix: A sparse R matrix
    :return: The corresponding full R matrix
    """
    func_string = 'as_matrix <- function(x){return(as.matrix(x))}'
    as_matrix = STAP(func_string, "as_matrix")
    return as_matrix.as_matrix(sparse_matrix)


def predict_glm(data, models_df):
    """
    :param data: The dataset dictionary.
    :param models_df: A dataframe of model weights. Column names of weights should correspond to those in the dataset.
    :return: 2D-array of predictions with dimension num_models x num_instances.
    """
    model_weights = models_df[data['variable_names']].values
    preds = np.sign(np.einsum("ik,jk->ij", model_weights, data['X']) - 1e-6)
    return preds


def get_accuracies(data, models_df):
    """
    :param data: The dataset dictionary.
    :param models_df: A dataframe of model weights. Column names of weights should correspond to those in the dataset.
    :return: 1D-array of length num_models reporting the accuracy of each model across the combined train & test data.
    """
    preds = predict_glm(data, models_df)
    return np.mean(preds == data['Y'], axis=1)


def run_cv_glmnet(data, cvindices, fold_id, family="binomial", glmnet_kwargs=None):
    """
    Runs glmnet with cross-validation and saves all models from each run.
    :param data: The data dictionary.
    :param cvindices: The cross-validation indices as a dictionary.
    :param fold_id: A string specifying the folds to use for cross-validation from the cv_indices dictionary.
    :param family: The response type. Passed to glmnet.
    :param glmnet_kwargs: Keyword arguments to pass to glmnet. See glmnet documentation for details.
    :return: pandas dataframe containing all models and their accuracies across the combined train/test data.
    """
    if not glmnet_kwargs:
        glmnet_kwargs = {}

    # pick lambda values to test
    models_df = fit_glm(data, family=family, glmnet_kwargs=glmnet_kwargs)
    models_df['fold'] = 0
    lambda_ = models_df['lambda_']
    glmnet_kwargs['lambda'] = FloatVector(lambda_)

    # fit crossfold models
    for fold in np.unique(cvindices[fold_id]):
        print("fitting fold %i..." % fold)
        assert np.issubdtype(int, np.integer), "CV indices must be integers"
        data_split = split_data_by_cvindices(deepcopy(data), cvindices, fold_id=fold_id, fold_num_test=int(fold))
        fit_fold = fit_glm(data_split, family=family, glmnet_kwargs=glmnet_kwargs)
        fit_fold['fold'] = fold
        models_df = models_df.append(fit_fold)

    # get accuracy metrics
    models_df['acc'] = get_accuracies(data, models_df)

    return models_df
