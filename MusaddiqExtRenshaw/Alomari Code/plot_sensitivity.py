import numpy as np
import matplotlib.pyplot as plt
import colorsys
import pickle

dir_save = "/home/omari/Datasets/sensitivity/"
features = ["ECAI_colours", "ECAI_objects", "ECAI_faces", "ECAI_actions",
            "Baxter_shapes", "Baxter_colours", "Baxter_distances"]

N = len(features)
HSV_tuples = [(x * 1.0 / N, 1, 1) for x in range(N)]
RGB = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)

fig, axes = plt.subplots(nrows=1, ncols=2)


def _plot_f_score():
    # fig, ax = plt.subplots(1,2,1)
    ax = axes[0]
    data = {}
    for f, c in zip(features, RGB):
        x, y = pickle.load(open(dir_save + f + '_sensitivity.p', "rb"))
        # y = y/np.max(y)
        for x1, y1 in zip(x, y):
            if x1 not in data:
                data[x1] = []
            data[x1].append(y1)
        ax.plot(x, y, c=c)
    ax.grid(True, zorder=5)

    ax = axes[1]
    keys = sorted(data.keys())
    Y = []
    for k in keys:
        print k,
        Y.append(np.sum(data[k]) / N)

    ax.plot(keys, Y)
    y_max = np.max(Y)
    ax.plot([0, np.max(keys)], [y_max, y_max], '--r')
    k = keys[Y.index(y_max)]
    ax.plot([k, k], [0.0, 0.45], '--r')
    ax.set_ylim([0.2, 0.45])
    plt.show()


_plot_f_score()
