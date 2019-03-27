import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import attr
from attr.validators import instance_of

from calcium_bflow_analysis.dff_tools import locate_spikes_peakutils, scatter_spikes


@attr.s
class ShowLabeledAndUnlabeled:
    """
    Plots a simple comparison of the dF/F traces that
    originated from the unlabeled cells and the labeled
    cells.
    """

    foldername = attr.ib(validator=instance_of(pathlib.Path))
    fps = attr.ib(default=58.2, validator=instance_of(float))
    file_pairs = attr.ib(init=False)

    def run(self):
        """ Main pipeline """
        self.file_pairs = self._find_results_and_colabeled(self.foldername)
        labeled, unlabeled = self._stack_dff_arrays(self.file_pairs)
        self._plot_against(labeled, unlabeled, self.fps)

    def _find_results_and_colabeled(self, folder):
        """
        Populate a DataFrame with pairs of results filenames and the
        "colabeled_idx.npy" filenames, which contain the indices of
        the cells that are also labeled.
        """
        file_pairs = pd.DataFrame({"results": [], "colabeled": []})
        for result_file in folder.rglob("*results.npz"):
            try:
                name = result_file.name[:-11] + "colabeled_idx.npy"
                print(name)
                colabel_file = next(result_file.parent.glob(name))
            except StopIteration:
                continue
            else:
                pair = pd.Series({"results": result_file, "colabeled": colabel_file})
                file_pairs = file_pairs.append(pair, ignore_index=True)
        return file_pairs

    def _stack_dff_arrays(self, file_pairs):
        """
        Loads each pair of dF/F data and colabeled cells and
        separates the labeled and the unlabeled cells.
        """
        labeled_data = np.array([])
        unlabeled_data = np.array([])
        for idx, file in file_pairs.iterrows():
            all_data = np.load(file["results"])["F_dff"]
            labeled_idx = np.load(file["colabeled"])
            print(labeled_idx)
            labeled = all_data[labeled_idx, :]
            unlabeled = np.delete(all_data, labeled_idx)

            labeled_data = np.hstack((labeled_data, labeled))
            unlabeled_data = np.hstack(unlabeled_data, unlabeled)

        return labeled_data, unlabeled_data

    def _plot_against(self, labeled, unlabeled, fps):
        """
        Plot one against the other the labeled and unlabeled
        traces.
        """
        fig, ax = plt.subplots(1, 2, 1, sharey=True)
        spikes_labeled = locate_spikes_peakutils(labeled, fps=fps)
        spikes_unlabeled = locate_spikes_peakutils(unlabeled, fps=fps)
        x_ax = np.arange(labeled.shape[1]) / fps
        scatter_spikes(
            labeled, spikes_labeled, downsample_display=1, time_vec=x_ax, ax=ax[1]
        )
        scatter_spikes(
            unlabeled, spikes_unlabeled, downsample_display=1, time_vec=x_ax, ax=ax[0]
        )
        fig.suptitle("Comparison of PNN-negative neurons (left) and positive")


if __name__ == "__main__":
    foldername = pathlib.Path("/data/Amit_QNAP/ForHagai")
    showl = ShowLabeledAndUnlabeled(foldername, fps=58.2)
    showl.run()
    print(showl.file_pairs)

