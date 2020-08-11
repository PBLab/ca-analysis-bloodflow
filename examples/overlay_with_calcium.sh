#! /bin/bash

conda init bash
source ~/.bashrc
conda activate calcium

python "/data/MatlabCode/PBLabToolkit/CalciumDataAnalysis/python-ca-analysis-bloodflow/examples/side_by_side.py"

