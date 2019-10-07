import time
import dill
import numpy as np
import pandas as pd
from datetime import datetime

pd.set_option('display.max_columns', 30)
np.set_printoptions(precision = 2)
np.set_printoptions(suppress = True)

from eqm.paths import *
from eqm.data import *