This repository contains a software implementation of the methods described in:

[**Predictive Multiplicity in Classification**](https://arxiv.org/abs/1909.06677)    
Charlie Marx, Flavio Calmon, Berk Ustun

## Installation

#### Requirements

- Python 3
- CPLEX

#### CPLEX

CPLEX is cross-platform commercial optimization tool with a Python API. It is free for students and faculty at accredited institutions. To get CPLEX:

1. Register for [IBM OnTheHub](https://ibm.onthehub.com/WebStore/Account/VerifyEmailDomain.aspx)
2. Download the *IBM ILOG CPLEX Optimization Studio* from the [software catalog](https://ibm.onthehub.com/WebStore/ProductSearchOfferingList.aspx?srch=CPLEX)
3. Install the CPLEX Optimization Studio.
4. Setup the CPLEX Python API [as described here](https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/GettingStarted/topics/set_up/Python_setup.html).

If you have problems installing CPLEX, check the [CPLEX user manual](http://www-01.ibm.com/support/knowledgecenter/SSSA5P/welcome) or the [CPLEX forums](https://www.ibm.com/developerworks/community/forums/html/forum?id=11111111-0000-0000-0000-000000002059). 
 
## Usage

1. Preprocess the dataset you wish to use (a csv file in the `data` directory) by running `scripts/ex_create_dataset_files.py` to generate the pickle file for the data. 
2. For MIP solutions: Run `scripts/ex_solve_mip.py`, specifying the name of your dataset. 
3. For glmnet solutions: Run `ex_glmnet_cv.py`, specifying the name of your dataset. 
 
## Development

The code in this repository is currently under active development, and may therefore change substantially with each commit.   
   
## Reference

If you use our code, please cite [our paper](https://arxiv.org/abs/1909.06677).

```
@misc{marx2019predictive,
    title={Predictive Multiplicity in Classification},
    author={Charles T. Marx and Flavio du Pin Calmon and Berk Ustun},
    year={2019},
    eprint={1909.06677},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
