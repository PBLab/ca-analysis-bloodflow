import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


fname = "/data/Amit_QNAP/ForHagai/FOV3/355_GCaMP6-Ch2_WFA-590-Ch1_X25_mag3_act3b-940nm_256px_20180313_00001_CHANNEL_2_results.npz"

data = np.load(fname)
dff = data["F_dff"]
time_vec = np.arange(dff.shape[1]) / 58.2

fig, ax = plt.subplots()

# 4 = 2
# 3 = 0
# 2 = 9
# 1 = 6

data_for_figure = dff[[6, 9, 0, 2]]
data_for_figure = pd.DataFrame(data_for_figure.T).rolling(58).mean().to_numpy().T

ax.plot(
        time_vec,
        (data_for_figure + np.arange(data_for_figure.shape[0])[:, np.newaxis]).T,
        linewidth=2,
        alpha=0.8,
        )

ax.set_xlabel('Time (seconds)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.savefig('/data/Amit_QNAP/ForHagai/raw_figure_for_article.pdf')

plt.show()

