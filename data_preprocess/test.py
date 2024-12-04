import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

subjects = [1, 2, 3, 4, 5, 8, 9]
episodes = 10

def load_data(subject, episode):
    pos_path = f'datasets/UI-PRMD/Segmented Movements/Kinect/Positions/m{subject:02d}_s01_e{episode:02d}_positions.txt'

    pos_df = pd.read_csv(pos_path, header=None)
    return np.array(pos_df)

data_input = []
for subject in subjects:
    temp = np.zeros([100, 100])
    for episode in range(1, episodes + 1):
        temp = load_data(subject, episode)
        data_input.append(temp)

data_tot = []

for i in range(1, 7 + 1):
    for j in range(1, 10 + 1):
        plt.subplot(7, 10, (i-1) * 10 + j)
        plt.plot(data_input[(i-1) * 10 + j-1])

plt.cla()
plt.subplot(111)
plt.plot(data_input[10])
plt.show()