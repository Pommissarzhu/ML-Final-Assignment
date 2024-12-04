import pandas as pd
import numpy as np
from utils import *
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d


joints_number = 22
joint_dim = 3
frames_number = 63
joints_tree = [[0, 1], [0, 14], [0, 18],
                [1, 2], [2, 3], [3, 4], [3, 6], [3, 10],
                [4, 5],
                [6, 7], [7, 8], [8, 9],
                [10, 11], [11, 12], [12, 13],
                [14, 15], [15, 16], [16, 17],
                [18, 19], [19, 20], [20, 21]]

def cubic_interpolation(arr, target_rows):
    """
    使用 SciPy 的三次插值将二维数组的行数调整到目标行数。
    参数:
        arr: 原始二维 ndarray
        target_rows: 目标行数
    返回:
        调整后的二维 ndarray
    """
    original_rows, cols = arr.shape
    x_original = np.arange(original_rows)  # 原始行索引
    x_target = np.linspace(0, original_rows - 1, target_rows)  # 目标行索引

    # 创建目标数组
    interpolated_arr = np.zeros((target_rows, cols))

    # 对每一列进行三次插值
    for col in range(cols):
        cubic_interp_func = interp1d(x_original, arr[:, col], kind='cubic', fill_value="extrapolate")
        interpolated_arr[:, col] = cubic_interp_func(x_target)

    return interpolated_arr

def smooth_data(readpos, readang):
    size = readpos.shape[1]  # 获取列数
    readpos_sm = np.zeros_like(readpos)
    readang_sm = np.zeros_like(readang)

    for i in range(size):
        # 使用 Savitzky-Golay 滤波器进行平滑
        readpos_sm[:, i] = savgol_filter(readpos[:, i], window_length=5, polyorder=2)
        readang_sm[:, i] = savgol_filter(readang[:, i], window_length=20, polyorder=2, mode='interp')

    return readpos_sm, readang_sm


def raw_to_array(raw_data):
    new_data = np.zeros((raw_data.shape[0], 22, 3))
    for index, row in raw_data.iterrows():
        for i in range(joints_number):
            new_data[index, i, 0] = row[i * 3]
            new_data[index, i, 1] = row[i * 3 + 1]
            new_data[index, i, 2] = row[i * 3 + 2]
    return new_data

def draw_plot_norm(input_frame, joints_tree, ax):
    for i in joints_tree:
        x, y, z = [np.array([input_frame[i[0]][j], input_frame[i[1]][j]]) for j in range(3)]
        ax.plot(x, y, z, c='r')

    RADIUS = 1  # space around the subject
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

def load_data(subject, episode):
    pos_path = f'datasets/UI-PRMD/Incorrect Segmented Movements/Kinect/Positions/m07_s{subject:02d}_e{episode:02d}_positions_inc.txt'
    ang_path = f'datasets/UI-PRMD/Incorrect Segmented Movements/Kinect/Angles/m07_s{subject:02d}_e{episode:02d}_angles_inc.txt'
    joints_number = 22
    joint_dim = 3
    frames_number = 63
    joints_tree = [[0, 1], [0, 14], [0, 18],
                   [1, 2], [2, 3], [3, 4], [3, 6], [3, 10],
                   [4, 5],
                   [6, 7], [7, 8], [8, 9],
                   [10, 11], [11, 12], [12, 13],
                   [14, 15], [15, 16], [16, 17],
                   [18, 19], [19, 20], [20, 21]]

    ang_raw = pd.read_csv(ang_path, header=None)
    pos_raw = pd.read_csv(pos_path, header=None)

    skeleton_pos = raw_to_array(pos_raw)
    skeleton_ang = raw_to_array(ang_raw)

    # if skeleton_pos.shape[0] > 63:
    print(skeleton_pos.shape[0])
    #     skeleton_pos = skeleton_pos[:63]
    #     skeleton_ang = skeleton_ang[:63]

    # skeleton_pos, skeleton_ang = smooth_data(skeleton_pos, skeleton_ang)

    frames_number = skeleton_pos.shape[0]

    for i in range(frames_number):
        frame_pos = skeleton_pos[i, :, :]
        frame_ang = skeleton_ang[i, :, :]

        pi = np.pi

        # chest, neck, head
        rot_2 = eulers_2_rot_matrix(frame_ang[0, :] * pi / 180.0)
        frame_pos[1, :] = (rot_2 @ frame_pos[1, :].T).T + frame_pos[0, :]
        rot_3 = rot_2 @ eulers_2_rot_matrix(frame_ang[1, :] * pi / 180.0)
        frame_pos[2, :] = (rot_3 @ frame_pos[2, :].T).T + frame_pos[1, :]
        rot_4 = rot_3 @ eulers_2_rot_matrix(frame_ang[2, :] * pi / 180.0)
        frame_pos[3, :] = (rot_4 @ frame_pos[3, :].T).T + frame_pos[2, :]
        rot_5 = rot_4 @ eulers_2_rot_matrix(frame_ang[3, :] * pi / 180.0)
        frame_pos[4, :] = (rot_5 @ frame_pos[4, :].T).T + frame_pos[3, :]
        rot_6 = rot_5 @ eulers_2_rot_matrix(frame_ang[4, :] * pi / 180.0)
        frame_pos[5, :] = (rot_6 @ frame_pos[5, :].T).T + frame_pos[4, :]

        # left arm
        rot_7 = eulers_2_rot_matrix(frame_ang[2, :] * pi / 180.0)
        frame_pos[6, :] = (rot_7 @ frame_pos[6, :].T).T + frame_pos[2, :]
        rot_8 = rot_7 @ eulers_2_rot_matrix(frame_ang[6, :] * pi / 180.0)
        frame_pos[7, :] = (rot_8 @ frame_pos[7, :].T).T + frame_pos[6, :]
        rot_9 = rot_8 @ eulers_2_rot_matrix(frame_ang[7, :] * pi / 180.0)
        frame_pos[8, :] = (rot_9 @ frame_pos[8, :].T).T + frame_pos[7, :]
        rot_10 = rot_9 @ eulers_2_rot_matrix(frame_ang[8, :] * pi / 180.0)
        frame_pos[9, :] = (rot_10 @ frame_pos[9, :].T).T + frame_pos[8, :]

        # right arm
        rot_11 = eulers_2_rot_matrix(frame_ang[2, :] * pi / 180.0)
        frame_pos[10, :] = (rot_11 @ frame_pos[10, :].T).T + frame_pos[2, :]
        rot_12 = rot_11 @ eulers_2_rot_matrix(frame_ang[10, :] * pi / 180.0)
        frame_pos[11, :] = (rot_12 @ frame_pos[11, :].T).T + frame_pos[10, :]
        rot_13 = rot_12 @ eulers_2_rot_matrix(frame_ang[11, :] * pi / 180.0)
        frame_pos[12, :] = (rot_13 @ frame_pos[12, :].T).T + frame_pos[11, :]
        rot_14 = rot_13 @ eulers_2_rot_matrix(frame_ang[12, :] * pi / 180.0)
        frame_pos[13, :] = (rot_14 @ frame_pos[13, :].T).T + frame_pos[12, :]

        # left leg
        rot_15 = eulers_2_rot_matrix(frame_ang[0, :] * pi / 180.0)
        frame_pos[14, :] = (rot_15 @ frame_pos[14, :].T).T + frame_pos[0, :]
        rot_16 = rot_15 @ eulers_2_rot_matrix(frame_ang[14, :] * pi / 180.0)
        frame_pos[15, :] = (rot_16 @ frame_pos[15, :].T).T + frame_pos[14, :]
        rot_17 = rot_16 @ eulers_2_rot_matrix(frame_ang[15, :] * pi / 180.0)
        frame_pos[16, :] = (rot_17 @ frame_pos[16, :].T).T + frame_pos[15, :]
        rot_18 = rot_17 @ eulers_2_rot_matrix(frame_ang[16, :] * pi / 180.0)
        frame_pos[17, :] = (rot_18 @ frame_pos[17, :].T).T + frame_pos[16, :]

        # right leg
        rot_19 = eulers_2_rot_matrix(frame_ang[0, :] * pi / 180.0)
        frame_pos[18, :] = (rot_19 @ frame_pos[18, :].T).T + frame_pos[0, :]
        rot_20 = rot_19 @ eulers_2_rot_matrix(frame_ang[18, :] * pi / 180.0)
        frame_pos[19, :] = (rot_20 @ frame_pos[19, :].T).T + frame_pos[18, :]
        rot_21 = rot_20 @ eulers_2_rot_matrix(frame_ang[19, :] * pi / 180.0)
        frame_pos[20, :] = (rot_21 @ frame_pos[20, :].T).T + frame_pos[19, :]
        rot_22 = rot_21 @ eulers_2_rot_matrix(frame_ang[20, :] * pi / 180.0)
        frame_pos[21, :] = (rot_22 @ frame_pos[21, :].T).T + frame_pos[20, :]

        skeleton_pos[i, :, :] = frame_pos

    skel_pos_min = skeleton_pos.min(axis=(0, 1), keepdims=True)
    skel_pos_max = skeleton_pos.max(axis=(0, 1), keepdims=True)

    # skel_pos_norm = (skeleton_pos - skel_pos_min) / (skel_pos_max - skel_pos_min)
    skel_pos_norm = (skeleton_pos - skeleton_pos.min()) / (skeleton_pos.max() - skeleton_pos.min())

    skel = skel_pos_norm.reshape(skel_pos_norm.shape[0], -1)



    # fig = plt.figure()
    # plt.plot(skel)
    plt.subplot(131)
    plt.plot(skel_pos_norm[:, :, 0])
    plt.subplot(132)
    plt.plot(skel_pos_norm[:, :, 1])
    plt.subplot(133)
    plt.plot(skel_pos_norm[:, :, 2])
    # plt.show()

    return skel
    # ax = fig.add_subplot(111, projection='3d')
    # # draw_plot(frame, joints_tree, ax)
    #
    # plt.ion()
    # i = 0
    # while i < skel_pos_norm.shape[0]:
    #     # ax.lines = []
    #     ax.cla()
    #     # for line in ax.lines:
    #     #     line.remove()
    #
    #     draw_plot_norm(skel_pos_norm[i, :, :], joints_tree, ax)
    #     plt.pause(0.03)
    #     i += 1
    #     if i == skel_pos_norm.shape[0]:
    #         i = 0
    #
    # plt.ioff()
    # plt.show()
if __name__ == '__main__':
    subjects = [1, 2, 3, 4, 5, 8, 9]
    episodes = 10

    data_norm = []

    temp = np.zeros((63, 66))

    tot = 0

    hsyhsy = np.zeros((70, 75, 66))

    for subject in subjects:
        for episode in range(1, episodes + 1):
            temp = load_data(subject, episode)
            tot += temp.shape[0]
            temp = cubic_interpolation(temp, 75)
            data_norm.append(temp)

    plt.cla()

    for i in range(1, 7 + 1):
        for j in range(1, 10 + 1):
            plt.subplot(7, 10, (i - 1) * 10 + j)
            plt.plot(data_norm[(i - 1) * 10 + j - 1])
            hsyhsy[(i - 1) * 10 + j - 1, :, :] = data_norm[(i - 1) * 10 + j - 1]


    # plt.cla()
    # plt.subplot(111)
    # plt.plot(data_norm[10])
    plt.show()

    hsyhsy = hsyhsy.reshape(hsyhsy.shape[0], -1)

    hsy_df = pd.DataFrame(data=hsyhsy)
    # pd.DataFrame.to_csv(hsy_df, 'data_m07_incorrect.csv', header=False, index=False)

    # tot = 0
    # for item in data_norm:
    #     tot += item.shape[0]
    print(tot / 70.0)


