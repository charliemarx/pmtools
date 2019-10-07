

# automatically configure the repo directory path
# if this fails, hard code in the absolute path to the directory
# Ex: repo_dir = "/path/to/here/.../eqm/"
repo_dir = dirname(dirname(sys.frame(1)$ofile))

# set up paths
data_dir = file.path(repo_dir, 'data')
results_dir = file.path(repo_dir, 'results')
setwd(repo_dir)

# load helper functions
source("eqm/glmnet_experiments_helper.R")

# all datasets available
data_names = c('compas_arrest_small', 'compas_violent_small',
               'recidivism_CA_arrest', 'recidivism_NY_arrest',
               'recidivism_CA_drug', 'recidivism_NY_drug', 
               'pretrial_CA_arrest', 'pretrial_NY_arrest',
               'pretrial_CA_fta', 'pretrial_NY_fta')

#######################################
#############  DASHBOARD  #############
#######################################

data_run = data_names # c('compas_violent_small')
num_lambda = 100
alphas = seq(0, 1, 0.1)
thresholds = seq(0, 1, 0.5)

# if TRUE, will not only run datasets for which a results file does not exist
ONLY_RUN_MISSING = TRUE
RUN_ALL_PLOTS = TRUE

#######################################
#######################################

for (data_name in data_run) {
  # get the output filename
  outfile = file.path(repo_dir, "results", data_name, get_rdata_filename(data_name))
  flippedfile = get_flipped_plot_filename(data_name) 
  discrepancyfile = get_disc_plot_filename(data_name)
  coeffile = file.path(repo_dir, "results", data_name, get_coef_filename(data_name))
  
  cat(paste('Saving results to', outfile, '\n'))
  cat(paste('Saving flipped plots to', flippedfile, '\n'))
  cat(paste('Saving discrepancy plots to', discrepancyfile, '\n'))
  
  
  # decide whether to run glmnet, and whether to generate plots
  run_glmnet_flag = !file.exists(outfile) | !ONLY_RUN_MISSING
  run_plots_flag = !file.exists(outfile) | 
                   !file.exists(flippedfile) | 
                   !file.exists(discrepancyfile) | 
                   !ONLY_RUN_MISSING |
                    RUN_ALL_PLOTS
  
  if (run_glmnet_flag) {
    # run the experiment for this dataset
    bundle = run_cv_glmnet(data_name, 
                           num_lambda = num_lambda, 
                           alphas = alphas, 
                           thresholds = thresholds)
    
    # save data to disk and reload
    saveRDS(bundle, outfile)
  }
  
  # read the results from disk
  bundle = readRDS(outfile)
  
  # unpack the results
  results = bundle[["results"]]
  X_train = bundle[["X_train"]]
  Y_train = bundle[["Y_train"]]
  X_test = bundle[["X_test"]]
  Y_train = bundle[["Y_train"]]
  cv_indices = bundle[["cv_indices"]]
  
  # save the coefficients in a csv file
  coefs = filter_results_to_params(df=results)
  write.csv(coefs, coeffile, row.names=F)
    
  if (run_plots_flag) {
    # get useful epsilon benchmarks
    e_p = get_epsilon_p(df=results, p=0.05)
    e_se = get_epsilon_se(df=results)
    
    # get the multiplicity results
    plot_df = get_epsilon_level_sets(results, X_train, X_test)
    
    disc_plot_df = plot_df %>% select(pct_max_discrepancy_train, pct_max_discrepancy_test, epsilon) %>%
      gather(pct_max_discrepancy_train, pct_max_discrepancy_test, -epsilon, key="variable", value="value") %>%
      mutate(variable=replace(variable, variable=="pct_max_discrepancy_train", "Training Set")) %>% 
      mutate(variable=replace(variable, variable=="pct_max_discrepancy_test", "Test Set"))
    
    # generate plots
    # flipped = epsilon_plot(plot_df, x_name="epsilon", y_name="pct_flippable", x_lab="Error Tolerance", y_lab="Ambiguous Predictions", e_se=e_se, e_p=e_p)
    # discrepancy = epsilon_plot(plot_df, x_name="epsilon", y_name="pct_max_discrepancy_train", x_lab="Error Tolerance", y_lab="Maximum Discrepancy", e_se=e_se, e_p=e_p)
    
    # generate the plots
    ambiguity_plot = ggplot(plot_df, aes(x=epsilon, y=pct_flippable)) +
      geom_step(size=1.5, alpha=0.5, color="#00BFC4") + 
      geom_point(size=3, color="#00BFC4") +
      ylab("Ambiguous Predictions") + 
      xlab("Error Tolerance") +
      scale_y_continuous(labels=scales::percent_format(accuracy = 1), breaks=pretty_breaks(8)) +
      scale_x_continuous(labels=scales::percent_format(accuracy = 0.1), breaks=pretty_breaks(5)) +
      default.plot.theme()
    
    disc_plot = ggplot(disc_plot_df, aes(x=epsilon, y=value, color=variable)) +
      geom_step(size=1.5, alpha=0.5) + 
      geom_point(size=3) +
      ylab("Maximum Discrepancy") + 
      xlab("Error Tolerance") +
      scale_y_continuous(labels=scales::percent_format(accuracy = 1), breaks=pretty_breaks(8)) +
      scale_x_continuous(labels=scales::percent_format(accuracy = 0.1), breaks=pretty_breaks(5)) +
      default.plot.theme() +
      theme(legend.justification = c(0, 1), legend.position = c(0.03, 0.97))
      
    save_to_disk(ambiguity_plot, flippedfile)
    save_to_disk(disc_plot, discrepancyfile)
    
  }
}



 #############



  




