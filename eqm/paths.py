from pathlib import Path

# parent directory
repo_dir = Path(__file__).absolute().parent.parent

# directory containing data files
data_dir = repo_dir / 'data/'

# directory containing results
results_dir = repo_dir / 'results/'
paper_dir = repo_dir / 'paper/'

# create directories that don't exist
for f in [data_dir, results_dir, paper_dir]:
    f.mkdir(exist_ok = True)
