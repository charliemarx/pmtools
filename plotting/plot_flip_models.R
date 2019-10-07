

# automatically configure the repo directory path
# if this fails, hard code in the absolute path to the directory
# Ex: repo_dir = "/path/to/here/.../eqm/"
repo_dir = dirname(dirname(sys.frame(1)$ofile))

# set up paths
data_dir = file.path(repo_dir, 'data')
results_dir = file.path(repo_dir, 'results')
paper_dir = file.path(repo_dir, 'paper')
setwd(repo_dir)

# load helper functions
source("eqm/glmnet_experiments_helper.R")
source("plotting/plotting_helper.R")

output_dir = paper_dir
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

##########################################

data_names = c('compas_arrest_small', 'compas_violent_small',
               'pretrial_CA_arrest', 'pretrial_CA_fta',
               'pretrial_NY_arrest', 'pretrial_NY_fta',
               'recidivism_CA_arrest', 'recidivism_CA_drug',
               'recidivism_NY_arrest', 'recidivism_NY_drug')
# data_names = c('compas_arrest_small')
fold_id = "K05N01"

##########################################

get_plots = function(data_name, fold_id) {
    # get filenames of the results
    flipped_fname = paste(data_name, fold_id, "flipped_processed_results.csv", sep="_")
    flipped_fname = file.path(results_dir, data_name, flipped_fname)
    discrepancy_fname = paste(data_name, fold_id, "discrepancy_raw_results.csv", sep="_")
    discrepancy_fname = file.path(results_dir, data_name, discrepancy_fname)
    
    cat(paste0("Loading flipped results from: ", flipped_fname, "\n"))
    cat(paste0("Loading discrepancy results from: ", discrepancy_fname, "\n"))
    
    assert("Flipped results csv file does not exist.", file.exists(flipped_fname))
    assert("Discrepancy results csv file does not exist.", file.exists(discrepancy_fname))
    
    # read the results from disk
    flipped_df = read.csv(flipped_fname)
    discrepancy_df = read.csv(discrepancy_fname)
  
    # todo create dataframe for (n, n_pos, n_neg) on train/test or load dataset from R
    
    num_instances = discrepancy_df %>%
        slice(1) %>%
        transmute(n = total_agreement + total_discrepancy) %>%
        pull(n)
    
    ##### plotting the flipped results ####
    # split into different model types
    flipped = flipped_df %>% filter(model_type == 'flipped')
    alternative = flipped_df %>% filter(model_type == 'alternative')
    baseline = flipped_df %>% filter(model_type == 'baseline')
    
    baseline_train_error = baseline %>% pull(train_error)
    baseline_test_error = baseline %>% pull(test_error)

    # the results to show in the plot
    plot_df = flipped %>%
        mutate(n = n_pos + n_neg) %>%
        select(train_error, test_error, n) %>%
        arrange(train_error, test_error) %>%
        mutate(id = row_number() / n(),
               marginal_train_error = train_error - baseline_train_error)
               #marginal_test_error = test_error - baseline_test_error)
    
    # the results duplicated by number of instances per flipped point
    duplicated_plot_df = plot_df %>%
        uncount(n) %>%
        mutate(id = row_number() / n(), 
               n = 1,
               marginal_train_error = train_error - baseline_train_error)
               #marginal_test_error = test_error - baseline_test_error)
    
    # melt dataframes so we can plot multiple lines on one plot
    plot_df = plot_df %>%
        gather(train_error, test_error, -n, -id, key="variable", value="value") %>%
        mutate(variable=replace(variable, variable=="train_error", "Train Error")) %>% 
        mutate(variable=replace(variable, variable=="test_error", "Test Error"))

    duplicated_plot_df = duplicated_plot_df %>%
        gather(train_error, test_error, -n, -id, key="variable", value="value") %>%
        mutate(variable=replace(variable, variable=="train_error", "Train Error")) %>% 
        mutate(variable=replace(variable, variable=="test_error", "Test Error"))

    duplicated_plot_df_transpose = duplicated_plot_df %>%
      gather(marginal_train_error, -n, -id, key="variable", value="value")
      #mutate(value=replace(value, value=="train_error", "Train Error")) %>% 
      #mutate(value=replace(value, value=="test_error", "Test Error"))
    
    # add point at (0, 0) if no entry for error_tolerance=0
    if (!(0 %in% duplicated_plot_df_transpose$marginal_train_error)) {
      row = list(0.0, 1.0, "marginal_train_error", 0.0)
      names(row) = c("id", "n", "variable", "value")
      duplicated_plot_df_transpose = rbind(duplicated_plot_df_transpose, row)
    }
    #if (!(0 %in% duplicated_plot_df_transpose$marginal_test_error)) {
    #  row = list(0.0, 1.0, "marginal_test_error", 0.0)
    #  names(row) = c("id", "n", "variable", "value")
    #  duplicated_plot_df_transpose = rbind(duplicated_plot_df_transpose, row)
    #}    
    
    #### Flipped Error Path ####
    flipped_error_path = ggplot(plot_df, aes(x=id, y=value, color = variable)) +
        geom_step(alpha = 0.75, size=1.5) +
        geom_point(size = 3) +
        scale_x_continuous(name = "Error Tolerance", breaks = pretty_breaks(5), labels=scales::percent_format(accuracy = 0.1)) +
        scale_y_continuous(name = "Percent of Instances Flipped", breaks = pretty_breaks(8), labels=scales::percent_format(accuracy = 1)) +
        default.plot.theme() +
        theme(legend.position = "bottom")

    save_to_disk(p = flipped_error_path,
                 file_name = paste0(data_name, "_flipped_error_path.png"))
    
    #### Flipped Error Path (Duplicated) ####
    flipped_error_path_duplicated = ggplot(duplicated_plot_df, aes(x=id, y=value, color = variable)) +
        geom_step(alpha = 0.75, size=1.5) +
        geom_point(size=3) +
        scale_x_continuous(name = "Error Tolerance", breaks = pretty_breaks(5), labels=scales::percent_format(accuracy = 0.1)) +
        scale_y_continuous(name = "Percent of Instances Flipped", breaks = pretty_breaks(8), labels=scales::percent_format(accuracy = 1)) +
        default.plot.theme() +
        theme(legend.position = "bottom")

    save_to_disk(p = flipped_error_path_duplicated,
                 file_name = paste0(data_name, "_flipped_error_path_duplicated.png"))
    
    #### Flipped Error Path (Duplicated, Transpose) ####
    flipped_error_path_duplicated_transpose = ggplot(duplicated_plot_df_transpose, aes(x=value, y=id, color = variable)) +
      geom_step(alpha = 0.75, color="#00BFC4", size=1.5) +
      geom_point(color="#00BFC4", size=3) +
      scale_x_continuous(name = "Error Tolerance", breaks = pretty_breaks(5), labels=scales::percent_format(accuracy = 0.1), limits=c(0, NA)) +
      scale_y_continuous(name = "Ambiguous Predictions", breaks = pretty_breaks(8), labels=scales::percent_format(accuracy = 1)) +
      default.plot.theme()
      #theme(legend.position = "bottom")

      
    save_to_disk(p = flipped_error_path_duplicated_transpose,
                 file_name = paste0(data_name, "_flipped_error_path_duplicated_transpose.png"))
    
    #### plotting the discrepancy results
    
    disc_df = discrepancy_df %>%
        select(train_discrepancy, test_discrepancy, epsilon) %>%
        mutate_all(function(v) as.numeric(as.character(v))) %>%
        mutate(epsilon = epsilon / num_instances) %>%
        arrange(epsilon)
    
    # melt data to plot upperbound and lowerbound on same plot
    disc_plot_df = disc_df %>%
        gather(train_discrepancy, test_discrepancy, -epsilon, key="variable", val="value") %>%
        mutate(variable=replace(variable, variable=="train_discrepancy", "Training Set")) %>% 
        mutate(variable=replace(variable, variable=="test_discrepancy", "Test Set"))
    
    #### Disc vs Epsilon Plot ####
    
    disc_vs_epsilon_path = ggplot(disc_plot_df, aes(x=epsilon, y=value, color=variable)) +
        geom_step(size=1.5, alpha=0.75) +
        geom_point(size=3) +
        scale_x_continuous(name = "Error Tolerance", breaks = pretty_breaks(5), limits = c(0, NA), labels=scales::percent_format(accuracy = 0.1)) +
        scale_y_continuous(name = "Maximum Discrepancy", breaks = pretty_breaks(8), limits = c(0, NA), labels=scales::percent_format(accuracy = 1)) +
        # todo no point/
        default.plot.theme() +
        theme(legend.justification = c(0, 1), legend.position = c(0.03, 0.97))
    
    save_to_disk(p = disc_vs_epsilon_path,
                 file_name = paste0(data_name, "_disc_vs_epsilon_path.png"))
    
    
    # print(disc_vs_epsilon_path)
    # print(flipped_error_path_duplicated)
    
    #### Error vs Epsilon Plot ####
    # 
    # error_vs_epsilon_path = ggplot(error_plot_df, aes(x=epsilon, y=value, color=variable)) +
    #     geom_step() +
    #     geom_point() +
    #     scale_x_continuous(name = "Tolerance", breaks = pretty_breaks(8), limits = c(2, NA)) +
    #     scale_y_continuous(name = "Error Gap", breaks = pretty_breaks(8), limits = c(2, NA)) +
    #     default.plot.theme() +
    #     theme(legend.position = "bottom")
    # 
    # save_to_disk(p = error_vs_epsilon_path,
    #              file_name = paste0(data_name, "_error_vs_epsilon_path.png"))

}  



for (data_name in data_names) {
  get_plots(data_name = data_name, fold_id = fold_id)
}
