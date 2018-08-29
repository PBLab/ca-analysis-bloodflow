import numpy as np
import pathlib
from ca_analysis.find_colabeled_cells import ColabeledCells, TiffChannels
import pytest




class TestColabeled:
    fname = r'677_677_gain_sweeps_as_lines_good015_60p002_chan_1_stack.tif'
    col = ColabeledCells(tif=pathlib.Path(fname), result_file=pathlib.Path(fname),
                         activity_ch=TiffChannels.ONE, morph_ch=TiffChannels.TWO)
    
    struct_elem_ans = [(3, np.array([[1, 1, 1],
                                     [1, 1, 1],
                                     [1, 1, 1]], np.uint8)),
                        (2, np.array([[0, 0, 1, 0, 0],
                                      [0, 1, 1, 1, 0],
                                      [1, 1, 1, 1, 1],
                                      [0, 1, 1, 1, 0],
                                      [0, 0, 1, 0, 0]], np.uint8))]
    @pytest.mark.parametrize("radius,exp", struct_elem_ans)
    def test_struct_element(self, radius, exp):
        gen_mask = self.col._create_mask(radius)
        assert np.all(gen_mask == exp)


if __name__ == '__main__':
    pass
