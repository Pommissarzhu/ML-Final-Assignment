import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def restore_joint_value(pos, tree):
    new_pos = pos
    for item in tree:
        new_pos[:, item[1] * 3] += new_pos[:, item[0] * 3]
        new_pos[:, item[1] * 3 + 1] += new_pos[:, item[0] * 3 + 1]
        new_pos[:, item[1] * 3 + 2] += new_pos[:, item[0] * 3 + 2]
    return new_pos


def draw_plot(input_frame, joints_tree, ax):
    for i in joints_tree:
        x, y, z = [np.array([input_frame[i[0] * 3 + j], input_frame[i[1] * 3 + j]]) for j in range(3)]
        ax.plot(x, y, z, c='r')

    RADIUS = 300  # space around the subject
    # xroot, yroot, zroot = input_frame[0], input_frame[1], input_frame[2]
    xroot, yroot, zroot = 0, 0, 0

    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])
    ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    # ax.set_zlim3d([0, 2 * RADIUS + zroot])
    # ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")




pos_file_path = '../datasets/UI-PRMD/Movements/Kinect/Positions/m01_s01_positions.txt'
ang_file_path = '../datasets/UI-PRMD/Movements/Kinect/Angles/m01_s01_angles.txt'

pos_raw = pd.read_csv(pos_file_path, delim_whitespace=True, header=None)
ang_raw = pd.read_csv(ang_file_path, delim_whitespace=True, header=None)

position = np.array(pos_raw)

joints_tree = [[0, 1], [0, 14], [0, 18],
               [1, 2], [2, 3], [3, 4], [3, 6], [3, 10],
               [4, 5],
               [6, 7], [7, 8], [8, 9],
               [10, 11], [11, 12], [12, 13],
               [14, 15], [15, 16], [16, 17],
               [18, 19], [19, 20], [20, 21]]

frames = restore_joint_value(position, joints_tree)
# frames = position
# frames = normalization(frames)

frame = frames[0]

frame = normalization(frame)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# draw_plot(frame, joints_tree, ax)

plt.ion()
i = 0
while i < frames.shape[0]:
    # ax.lines = []
    ax.cla()
    # for line in ax.lines:
    #     line.remove()

    draw_plot(frames[i], joints_tree, ax)
    plt.pause(0.03)
    i += 1
    if i == frames.shape[0]:
        i = 0

plt.ioff()
plt.show()
