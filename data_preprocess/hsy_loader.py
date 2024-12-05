import numpy as np
import pandas as pd


data_raw = pd.read_csv('../data_m07_correct.csv', header=None)
data_inc_raw = pd.read_csv('../data_m07_incorrect.csv', header=None)

data = np.array(data_raw).reshape((70, 75, 22, 3))
data_inc = np.array(data_inc_raw).reshape((70, 75, 22, 3))