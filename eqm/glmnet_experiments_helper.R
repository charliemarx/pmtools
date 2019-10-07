# set up paths

repo_dir <- dirname(dirname(sys.frame(1)$ofile))
data_dir = file.path(repo_dir, 'data')
paper_dir = file.path(repo_dir, 'paper')
results_dir = file.path(repo_dir, 'results')
setwd(repo_dir)

# install and load dependencies
dependencies = c('testit', 'plyr', 'tidyr', 'magrittr', 'dplyr', 'glmnet', 'pROC', 'ggplot2', 'scales', 'gridExtra')
not_installed <- dependencies[!(dependencies %in% installed.packages()[,"Package"])]
if(length(not_installed)) install.packages(not_installed)
lapply(dependencies, library, character.only = TRUE)

# set the random seed
set.seed(1337)

# generates the absolute filepath for a dataset
get_data_path = function(data_name) {
    data_file = paste0(data_name, '_processed.csv')
    data_file = file.path(data_dir, data_file)
    return(data_file)
}

get_results_filename = function(data_name) {
    return(paste0(data_name, '_glmnet_results.csv'))
}

get_rdata_filename = function(data_name) {
  return(paste0(data_name, '_glmnet_results.rds'))
}

get_disc_plot_filename = function(data_name) {
  return(paste0(data_name, '_glmnet_disc_plot.png'))
}

get_flipped_plot_filename = function(data_name) {
  return(paste0(data_name, '_glmnet_flipped_plot.png'))
}

get_coef_filename = function(data_name, fold_id="K05N01") {
  return(paste(data_name, fold_id, "glmnet_coefficients.csv", sep="_"))
}

# generates cross-validation indices given the number of instances and number of folds to use.
get_cv_idxs = function(n_instances, n_folds) {
    fold_size = floor(n_instances / n_folds)
    cv_idxs = rep(1:n_folds, fold_size)
    remainder = sample(1:n_folds, n_instances %% n_folds)
    cv_idxs = c(cv_idxs, remainder)
    cv_idxs = sample(cv_idxs, n_instances)
    return(cv_idxs)
}


#### generates train indices
# :param num: number of instances used for training
# :param frac: fraction of data used for training. Ignored if num is specified.
get_train_idxs = function(n_instances, frac=NULL, num=NULL) {
    assert(!is.null(num) | !is.null(frac))
    if (!is.null(num)) {
        return(sample(1:n_instances, num))
    }
    else {
        return(sample(1:n_instances, floor(n_instances * frac)))
    }
}


# computes various error metrics given the vectors of true and predicted outcomes.
get_metrics = function(y, scores, threshold=0.5) {

    y = as.numeric(y)
    scores = as.numeric(scores)

    # compute predictions of the model from the scores assigned to each point
    yhat = 2 * (scores >= threshold) - 1

    # get basic performance metrics
    tp = sum((y == 1) & (yhat == 1))
    fp = sum((y == -1) & (yhat == 1))
    tn = sum((y == -1) & (yhat == -1))
    fn = sum((y == 1) & (yhat == -1))

    # count instances
    n = length(y)
    n_pos = sum(y == 1)
    n_neg = sum(y == -1)

    # get secondary performance metrics
    tpr = tp / (tp + fn)
    fpr = fp / (tn + fp)
    error = (fp + fn) / n
    error_pos = fn / n_pos
    error_neg = fp / n_neg

    # approximate auc
    roc1 = roc(y, scores, quiet=T)
    auc1 = auc(roc1)
    #auc = mean(sample(scores[y == 1],100000,replace=T) > sample(scores[y == -1],100000,replace=T))

    # bundle metrics into vector
    metrics = c(tpr, fpr, auc1, error, n, n_pos, n_neg, error_pos, error_neg)
    names(metrics) = c('tpr', 'fpr', 'auc', 'error', 'n', 'n_pos', 'n_neg', 'error_pos', 'error_neg')

    return(metrics)
}


# computes SOFT predictions for multiple linear models on a dataset
# :param params: (num_feats x num_models) matrix of model parameters
# :param x: (num_instances x num_feats) matrix of data
# :returns: (num_instances x num_models) matrix of soft predictions in the interval [0, 1]
score_glm = function(params, x) {
    params = as.matrix(params)
    x = as.matrix(x)
    logits = x %*% params
    scores = exp(logits) / (1 + exp(logits))
    return(scores)
}


# computes HARD predictions for multiple linear models on a dataset
# :param params: (num_feats x num_models) matrix of model parameters
# :param x: (num_instances x num_feats) matrix of data
# :param threshold: The score threshold above which to predict the positive class
# :returns: (num_instances x num_models) matrix of hard predictions in the set {-1, 1}
predict_glm = function(params, x, threshold=0.5) {
    scores = score_glm(params, x)
    return(scores >= threshold)
}


# runs cross validation for glmnet
run_cv_glmnet = function(data_name, train_frac=0.7,
                         n_train_folds=5, thresholds=seq(0, 1, 0.1),
                         alphas=seq(0, 1, 0.1), num_lambda=50, save_to_disk=FALSE) {

    # setup paths
    output_dir = file.path(results_dir, data_name)
    output_file = file.path(output_dir, get_results_filename(data_name))

    # print logs
    cat(paste0('Fitting models for ', data_name, '.\n'))
    if (save_to_disk) {
        cat(paste('Saving results_file to', output_file, '\n'))
    }

    # read the data
    data_path = get_data_path(data_name)
    data = read.csv(data_path)

    # drop row_id if it is included
    if ('row_id' %in% colnames(data)) {
        data %<>% select(-row_id)
    }

    # split X and Y from data
    X = data[, 2:ncol(data)]
    X = cbind(rep(1, nrow(X)), X)
    names(X)[1] = '(Intercept)'
    Y = data[, 1]

    # oversample minority class
    minority_label = -sign(sum(Y))
    if (minority_label) {
        minority_idxs = which(Y == minority_label)
        majority_idxs = which(Y != minority_label)

        diff = length(majority_idxs) - length(minority_idxs)
        rep_idxs = sample(minority_idxs, diff, replace=T)

        shuffle_idxs = sample(c(minority_idxs, majority_idxs, rep_idxs))
        X = X[shuffle_idxs, ]
        Y = Y[shuffle_idxs]
    }

    # split train and test
    train_idxs = get_train_idxs(nrow(data), frac=train_frac)

    X_train = as.matrix(X[train_idxs, ])
    X_test = as.matrix(X[-train_idxs, ])

    Y_train = as.matrix(Y[train_idxs])
    Y_test = as.matrix(Y[-train_idxs])

    assert(nrow(X_train) == length(Y_train))
    assert(nrow(X_test) == length(Y_test))

    # get splits for training set cv folds
    train_cv_idxs = get_cv_idxs(nrow(X_train), n_train_folds)

    # ensure there is no variable named 'results_df' so we can use its existence to check if it is first loop iteration later
    if (exists('results_df')) {rm("results_df", pos=".GlobalEnv")}

    cat("Training fold 0\n")

    # get lambda values to use for all glmnet models for consistency
    out = glmnet(x=X_train, y=Y_train, intercept=FALSE, alpha=0.5, nlambda=num_lambda)
    lambda = out$lambda
    m = length(lambda)

    # initialize parameter dataframe to save the models in.
    # params = data.frame(c("alpha", "lambda", "threshold", "holdout_fold", paste0("w_", 1:nrow(X))))

    for (alpha in alphas) {
        # run glmnet on all train data
        out = glmnet(x=X_train, y=Y_train, intercept=FALSE, alpha=alpha, lambda=lambda)

        # compute soft predictions on train, test
        train_scores = score_glm(params=out$beta, x=X)
        test_scores = score_glm(params=out$beta, x=X_test)

        for (thresh in thresholds) {
            # compute error metrics
            train_metrics = apply(train_scores, 2, get_metrics, y=Y, threshold=thresh)
            test_metrics = apply(test_scores, 2, get_metrics, y=Y_test, threshold=thresh)
            metrics = t(cbind(train_metrics, test_metrics))
            rownames(metrics) = NULL

            # get model parameters to match the metrics matrix
            params = t(as.matrix(out$beta))
            params = rbind(params, params)
            colnames(params) = paste0("weight__", colnames(params))

            # save results to df
            holdout_fold = 0
            stat_type = c(rep('train', m), rep('test', m))
            results = as.data.frame(cbind(data_name, holdout_fold, alpha, lambda, thresh, stat_type, metrics, params))
            if (exists("results_df")) {results_df = rbind(results_df, results)}
            else {results_df = results}
        }
    }

    # run cv glmnet
    for (holdout_fold in sort(unique(train_cv_idxs))) {
        cat(paste('Training fold', holdout_fold, '\n'))

        # get training data
        X = X_train[train_cv_idxs != holdout_fold, ]
        Y = Y_train[train_cv_idxs != holdout_fold]

        # get validation data
        X_validation = X_train[train_cv_idxs == holdout_fold, ]
        Y_validation = Y_train[train_cv_idxs == holdout_fold]

        # convert to matrices
        X = as.matrix(X); Y = as.matrix(Y)
        X_validation = as.matrix(X_validation); Y_validation = as.matrix(Y_validation)

        for (alpha in alphas) {
            # get the models for this fold
            out = glmnet(x=X, y=Y, intercept=FALSE, alpha=alpha, lambda=lambda)

            # compute soft predictions on train, validation, test
            train_scores = score_glm(params=out$beta, x=X)
            val_scores = score_glm(params=out$beta, x=X_validation)
            test_scores = score_glm(params=out$beta, x=X_test)

            # get model parameters to match the metrics matrix
            params = t(as.matrix(out$beta))
            params = rbind(params, params, params)
            colnames(params) = paste0("weight__", colnames(params))

            # compute error metrics
            for (thresh in thresholds) {
                train_metrics = apply(train_scores, 2, get_metrics, y=Y, threshold=thresh)
                val_metrics = apply(val_scores, 2, get_metrics, y=Y_validation, threshold=thresh)
                test_metrics = apply(test_scores, 2, get_metrics, y=Y_test, threshold=thresh)
                metrics = t(cbind(train_metrics, val_metrics, test_metrics))
                rownames(metrics) = NULL

                # save results to df
                stat_type = c(rep('train', m), rep('validation', m), rep('test', m))
                fold_df = as.data.frame(cbind(data_name, holdout_fold, alpha, lambda, thresh, stat_type, metrics, params))

                # append the fold results to the total results
                results_df = rbind(results_df, fold_df)
            }
        }
    }

    # set column types
    ftr_to_num = c('auc', 'lambda', 'thresh', 'alpha', 'error', 'n',
                   'n_pos', 'n_neg', 'error_pos', 'error_neg', 'tpr', 'fpr')
    results_df[ftr_to_num] %<>% lapply(function(x) as.numeric(as.character(x)))

    # add model_id column
    r = results_df
    results_df$model_id = paste(r$data_name, "a", r$alpha, "lam", formatC(r$lambda, format = "e", digits = 3), "t", r$thresh, sep="_")

    # save results
    if (save_to_disk) {
        write.csv(results_df, file=output_file, row.names=F)
        cat(paste0('Saved results files for ', data_name, '.\n'))
    }

    cat(paste0(strrep('-', 30), '\n'))

    bundle = list(results_df, X_train, Y_train, X_test, Y_test, train_cv_idxs)
    names(bundle) = c("results", "X_train", "Y_train", "X_test", "Y_test", "cv_indices")

    return(bundle)
}



# runs cross validation for glmnet
run_cv_glmnet_multiple = function(data_names, train_frac=0.7,
                                  n_train_folds=5, thresholds=seq(0, 1, 0.1),
                                  alphas=seq(0, 1, 0.1), num_lambda=50, save_to_disk=TRUE) {
    for (data_name in data_names) {
        run_cv_glmnet(data_name=data_name, train_frac=train_frac,
                      n_train_folds=n_train_folds, thresholds=thresholds,
                      alphas=alphas, num_lambda=num_lambda, save_to_disk=save_to_disk)
    }
}


get_epsilon_p = function(df, p=0.05) {

    # get models summary
    s = summarize_models(df)

    # find all models with near-optimal test error
    best_test_error = min(s$test_error)
    e_p_optimal = s %>% filter(test_error <= best_test_error*(1 + p))

    # find the maximum training error gap among those models
    e_p = max(e_p_optimal$train_error_mean) - min(e_p_optimal$train_error_mean)

    return(e_p)
}


get_epsilon_se = function(df) {

    # get models summary
    s = summarize_models(df)

    # find all models within 1 SE of optimal validation error
    val_error_optimal_mean = min(s$val_error_mean)
    val_error_optimal_se = with(s, val_error_se[val_error_mean == val_error_optimal_mean])[1]
    e_es_optimal = s %>% filter(val_error_mean <= val_error_optimal_mean + val_error_optimal_se)

    # find the maximum training error gap among those models
    e_es = max(e_es_optimal$train_error_mean) - min(e_es_optimal$train_error_mean)

    return(e_es)
}


summarize_models = function(df) {
    # Computes error metrics for each set of parameters (threshold, alpha, lambda)
    # and returns a dataframe where each set of parameters only appears in a single row, with its relevant error metrics
    #
    # Train error: Average train error when performing cross-validation
    # Validation error: Average validation error when performing cross-validation
    # Test error: Test error of model trained on full train set (no validation set used)

    # compute the relevant statistics from each group
    # df %>% select(alpha, lambda) %>% distinct()
    train_stats = df %>%
        filter(stat_type == "train", holdout_fold != 0) %>%
        group_by(alpha, lambda, thresh) %>%
        summarize(train_error_mean = mean(error),
                  train_error_se = sd(error)) %>%
        ungroup()

    val_stats = df %>%
        filter(stat_type == "validation", holdout_fold != 0) %>%
        group_by(alpha, lambda, thresh) %>%
        summarize(val_error_mean = mean(error),
                  val_error_se = sd(error)) %>%
        ungroup()

    test_stats = df %>%
        filter(stat_type == "test", holdout_fold == 0) %>%
        group_by(alpha, lambda, thresh) %>%
        summarize(test_error = mean(error)) %>%
        ungroup()

    # recombine the useful statistics
    summary_df = train_stats %>%
        left_join(val_stats, by = c("alpha", "lambda", "thresh")) %>%
        left_join(test_stats, by = c("alpha", "lambda", "thresh")) %>%
        ungroup() %>%
        arrange(alpha, lambda, thresh)

    return(summary_df)
}



filter_results_to_params = function(df, params_tag = "weight__") {
    # Given the results dataframe, returns only the columns corresponding the weights of the models

    df = as.data.frame(df)
    param_names = startsWith(colnames(df), params_tag)
    params = df[, param_names] %>%
        mutate_all(function (x) as.numeric(as.character(x)))

    return(params)
}



filter_to_best_threshold = function(df) {

    # Given the results dataframe, deletes all models with a threshold that does not optimize validation error
    cv_results = df %>%
        filter(stat_type == "validation") %>%
        group_by(alpha, lambda, thresh) %>%
        summarize(validation_error_mean = mean(error)) %>%
        group_by(alpha, lambda) %>%
        slice(which.min(validation_error_mean)) %>%
        ungroup() %>%
        select(-validation_error_mean)

    flat_results = df %>% inner_join(cv_results, by = c("alpha", "lambda", "thresh"))
    #flat_results %>% distinct(alpha, lambda, stat_type, holdout_fold) %>% group_by(alpha, lambda) %>% tally()
    return(flat_results)
}



get_predictions = function(df, X, param_tag = "weight__") {
    # Given models and a dataset, computes the predictions of the models
    # :param results_df: dataframe (# models x # features) of trained models
    # :param x: dataset (# instances x # features) to compute predictions on
    # :returns: (# models x # instances) matrix of predictions

    # extract parameters from the results dataframe
    coefs = df %>%
        select(starts_with(param_tag)) %>%
        mutate_if(is.factor, function(v) as.numeric(as.character(v)))

    cnames = colnames(coefs)
    colnames(coefs) = gsub(param_tag, "", cnames)
    coefs =  coefs %>%
        as.matrix() %>%
        t()

    # check that the parameters match the data
    assert(all(colnames(coefs) == colnames(X)))

    # compute the predictions
    thresholds = df %>%
        select(thresh) %>%
        as.matrix() %>%
        as.numeric()

    probabilities = score_glm(coefs, X)
    predictions = apply(probabilities, MARGIN = 1, FUN = function(v) v > thresholds)  + 0
    return(predictions)
}



get_epsilon_level_sets = function(results_df, x_train, x_test, epsilons = seq(0, 0.1, 0.001)) {

    # get number of instances
    n_instances = nrow(x_train)
    n_instances_test = nrow(x_test)

    # make sure thresholds are chosen
    selected_df = filter_to_best_threshold(results_df)

    # get the best model
    summary_df = summarize_models(selected_df) %>%
        mutate(idx = row_number()) %>%
        select(idx, everything())


    best_summary_df = summary_df %>% slice(which.min(train_error_mean))
    best_train_error = best_summary_df %>% pull(train_error_mean)

    # get the predictions of each model on the train set
    train_results = selected_df %>%
        filter(stat_type == "train", holdout_fold == 0) %>%
        arrange(alpha, lambda, thresh) %>%
        mutate(idx = row_number()) %>%
        select(idx, everything()) %>%
        arrange(alpha, lambda, thresh)

    train_preds = get_predictions(train_results, x_train)
    test_preds = get_predictions(train_results, x_test)
    
    # the (num_models x num_instances) matrix of booleans of whether each model predicts each instance different than the optimal model
    best_results = train_results %>% inner_join(best_summary_df, by = c("alpha", "lambda", "thresh"))
    best_preds_train = get_predictions(best_results , x_train) %>% as.numeric()
    best_preds_test = get_predictions(best_results , x_test) %>% as.numeric()
    best_preds_train_matrix = matrix(best_preds_train, nrow = nrow(train_preds), ncol= length(best_preds_train), byrow = TRUE)
    best_preds_test_matrix = matrix(best_preds_test, nrow = nrow(test_preds), ncol= length(best_preds_test), byrow = TRUE)
    discrepancy_matrix_train = (train_preds != best_preds_train_matrix) + 0
    discrepancy_matrix_test = (test_preds != best_preds_test_matrix) + 0
    
    # for each value of epsilon, find the number of epsilon optimal models by training error
    m = length(epsilons)
    num_eps_equiv_models = rep(NA, m)
    max_eps_discrepancies_train = rep(NA, m)
    max_eps_discrepancies_test = rep(NA, m)
    num_unstable_preds = rep(NA, m)

    for (i in 1:m) {

        level_set_width = best_train_error + epsilons[i]
        level_set_idx = summary_df %>%
            filter(train_error_mean <= level_set_width) %>%
            pull(idx)

        num_eps_equiv_models[i] = length(level_set_idx)
        discrepancy_values_train = discrepancy_matrix_train[level_set_idx, ]
        discrepancy_values_test = discrepancy_matrix_test[level_set_idx, ]
        if (length(level_set_idx) > 1){
            max_eps_discrepancies_train[i] = max(rowSums(discrepancy_values_train))
            max_eps_discrepancies_test[i] = max(rowSums(discrepancy_values_test))
            num_unstable_preds[i] = sum(colSums(discrepancy_values_train) > 0)
        } else {
            max_eps_discrepancies_train[i] = 0
            max_eps_discrepancies_test[i] = 0
            num_unstable_preds[i] = 0
        }
    }
    eps_results = as.data.frame(cbind(epsilons, 
                                      num_eps_equiv_models, 
                                      num_unstable_preds, 
                                      num_unstable_preds / n_instances,  
                                      max_eps_discrepancies_train, 
                                      max_eps_discrepancies_train / n_instances,
                                      max_eps_discrepancies_test, 
                                      max_eps_discrepancies_test / n_instances_test))
    colnames(eps_results) = c("epsilon",
                              "num_equivalent",
                              "num_flippable",
                              "pct_flippable",
                              "max_discrepancy_train",
                              "pct_max_discrepancy_train",
                              "max_discrepancy_test",
                              "pct_max_discrepancy_test")
    return(eps_results)
}


epsilon_plot = function(plot_df, x_name, y_name, e_se=NA, e_p=NA, show_e=FALSE, x_lab="Error Gap", y_lab="NO Y-AXIS LABEL SPECIFIED", size_name=NA) {
  p = ggplot(plot_df, aes(x=as.numeric(plot_df[, x_name]), y=as.numeric(plot_df[, y_name])))
  p = p + geom_step(color="#00BFC4", size=1.5, alpha=0.5) + geom_point(color="#00BFC4", size=3)
  p = p + ylab(y_lab) + xlab(x_lab)
  p = p + scale_y_continuous(labels=scales::percent_format(accuracy = 1), breaks=pretty_breaks(8))
  p = p + scale_x_continuous(labels=scales::percent_format(accuracy = 0.1), breaks=pretty_breaks(5))
  # if (!is.na(size_name)) {p = p + geom_point(aes(size=plot_df[size_name]))}
  # else {p = p + geom_point(color="#00BFC4")}

  # format axis tick and title size
  p = p + default.plot.theme()

  # find x scale for formatting annotations
  xdata = plot_df[, x_name]
  xrange = xdata %>% max() - xdata %>% min()

  if (show_e) {
    if (e_p <= e_se) {min_name = expression(epsilon[p]); min_e = e_p; max_name = expression(epsilon[se]); max_e = e_se}
    else {min_name = expression(epsilon[se]); min_e = e_se; max_name = expression(epsilon[p]); max_e = e_p}

    label_height = 0.98 * max(plot_df[y_name])
    p = p + geom_vline(xintercept=min_e, color="red") + annotate("text", min_e - 0.03 * xrange, label_height, vjust = 1, label = min_name)
    p = p + geom_vline(xintercept=max_e, color="blue") + annotate("text", max_e + 0.03 *xrange, label_height, vjust = 1, label = max_name)
  }
  return(p)
}


default.plot.theme = function(){
  
  line_color = "#E9E9E9";
  
  default_theme = theme_bw() +
    theme(title = element_text(size = 18),
          plot.margin = margin(t = 0.25, r = 0.25, b = 0.25, l = 0.25, unit = "cm"),
          axis.line = element_blank(),
          panel.border = element_rect(size = 2.0, color = line_color),
          panel.grid.minor = element_blank(),
          panel.grid.major = element_line(linetype="solid", size=1.0, color=line_color),
          #
          axis.title.x = element_text(size = 32, margin = margin(t = 12, unit = "pt")),
          axis.text.x   = element_text(size = 28),
          axis.ticks.x  = element_line(size = 1.0, color = line_color),
          #
          axis.title.y = element_text(size = 32, margin = margin(r = 12, unit = "pt")),
          axis.text.y   = element_text(size=28),
          axis.ticks.y	= element_line(size=1.0, color = line_color),
          #
          legend.position="none",
          legend.title = element_blank(),
          legend.text = element_text(face="plain",size=28,angle=0,lineheight=30),
          #legend.key.width = unit(1.5, "cm"),
          #legend.key.height = unit(1.5, "cm"),
          #legend.text.align = 0,
          legend.margin = margin(t = 0, r = 0, b = 0, l = 0, unit = "cm"))
  
  return(default_theme);
}




save_to_disk = function(p, file_name){
  ggsave(plot = p, filename = file.path(paper_dir, file_name), dpi=100, width = 14, height = 7, units ="in")
}




