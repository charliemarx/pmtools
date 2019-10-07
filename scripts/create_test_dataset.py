from scripts.reporting import *
from eqm.cross_validation import generate_stratified_cvindices, filter_data_to_fold

# dashboard
data_name = 'adult_binarized'
random_seed = 1337
fold_id = 'K05N01'
fold_num = 1
np.random.seed(seed = random_seed)

## load data from disk
data_file_name = '%s_processed.csv' % data_name
data_file_name = data_dir / data_file_name
data, _ = load_processed_data(file_name = data_file_name.with_suffix('.pickle'))

#
# # keep only subset of variables
# to_keep = ['(Intercept)',
#            'race_is_causasian',
#            'race_is_african_american',
#            'race_is_hispanic',
#            'age_leq_25',
#            'age_25_to_45',
#            'age_geq_46',
#            'female',
#            'n_priors_eq_0',
#            'n_priors_geq_1',
#            'n_juvenile_felonies_eq_0',
#            'n_juvenile_felonies_geq_1',
#            'n_juvenile_misdemeanors_eq_0',
#            'n_juvenile_misdemeanors_geq_1',
#            'charge_degree_eq_M']
# to_drop = list(set(data['variable_names']) - set(to_keep))
#
# for n in to_drop:
#     data = remove_variable(data, n)

# compress dataset into distinct feature vectors and counts
U, x_to_u_idx, u_to_x_idx,  N = np.unique(data['X'], axis = 0, return_inverse = True, return_index = True, return_counts = True)

n_points = U.shape[0]
N_pos, N_neg = np.zeros(n_points), np.zeros(n_points)
for k in range(n_points):
    y = data['Y'][np.isin(u_to_x_idx, k)]
    N_pos[k], N_neg[k] = np.sum(y == 1), np.sum(y == -1)
count_df = pd.DataFrame({'n_pos': N_pos, 'n_neg': N_neg, 'idx': x_to_u_idx})

df = pd.concat((
    count_df.query('n_pos > 0').query('n_neg > 0').sample(50),
    count_df.query('n_pos > 0').query('n_neg == 0').sample(min(25, (np.logical_and(N_pos > 0, N_neg == 0)).sum())),
    count_df.query('n_pos == 0').query('n_neg > 0').sample(min(25, (np.logical_and(N_pos == 0, N_neg > 0)).sum()))
    ))

df = df.drop_duplicates()

n_pos = df['n_pos'].values.astype(int)
n_neg = df['n_neg'].values.astype(int)
n_counts = np.concatenate((n_pos, n_neg))
X = data['X'][df['idx'].values, 1:]

X_pos = np.repeat(X, n_pos, axis = 0)
X_neg = np.repeat(X, n_neg, axis = 0)
X_final = np.vstack((X_pos, X_neg))
Y_final = np.concatenate((np.ones(n_pos.sum()), -np.ones(n_neg.sum())))

# create data csv
data_df = pd.concat((pd.Series(Y_final), pd.DataFrame(X_final)), axis = 1)
data_df.columns = [data['outcome_name']] + data['variable_names'][1:]
output_file_name = 'test_%s_processed.csv' % data_name
output_file_name = data_dir / output_file_name
data_df.to_csv(output_file_name, header = True, index = False)


# reload dataset
data = load_data_from_csv(output_file_name)
assert data['X'].shape[0] == data_df.shape[0]
data = oversample_minority_class(data, random_state = random_seed)
cvindices = generate_stratified_cvindices(X = data['X'], strata = data['Y'], total_folds_for_cv = [3], total_folds_for_inner_cv = [5], replicates = 1, seed = random_seed)

# save to disk
save_data(file_name = output_file_name.with_suffix('.pickle'), data = data, cvindices = cvindices, overwrite = True, stratified = True, check_save = True)
save_data(file_name = output_file_name.with_suffix('.RData'), data = data, cvindices = cvindices, overwrite = True, stratified = True, check_save = True)
print('saved: %s' % output_file_name)