from eqm.data import *
from eqm.paths import *
from eqm.glmnet import run_cv_glmnet


# load data from disk
data_name = 'recidivism_arrest'
data_file_name = '%s/%s_processed.csv' % (data_dir, data_name)
data_file_name = Path(data_file_name)
random_seed = 1337
np.random.seed(seed=random_seed)

# load data from disk
data, cvindices = load_processed_data(file_name=data_file_name.with_suffix('.pickle'))

models = run_cv_glmnet(data, cvindices, fold_id='K05N03', glmnet_kwargs={'nlambda': 10, 'alpha': 0.5})
print(models)
print("Maximum accuracy:", models['acc'].max())