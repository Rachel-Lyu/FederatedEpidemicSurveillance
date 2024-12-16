#!/bin/bash
source ~/.bashrc
conda activate struct
python semiSyn.py --growthRateThr 30 --pval_thr 0.05 --nLoops 50 --maxPthr 1 --dir_name /net/dali/home/mscbio/rul98/semiSyn
