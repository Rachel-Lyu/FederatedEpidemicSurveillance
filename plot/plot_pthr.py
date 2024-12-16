import pandas as pd
import numpy as np
from scipy.special import ndtri
import scipy.stats as stats
import matplotlib.pyplot as plt
import scipy.stats as stats
import argparse
import os

dir_name = 'test_bs_pthr'
itv = 20
kLis = np.arange(100, 1000+itv, itv)
for share in [0.6, 0.8, 0.9, 2.0, 3.0, 5.0, 8.0, 10.0]:
    plt.figure(figsize = (20, 12))
    if share < 1:
        methLis = [['Stouffer', 'unweighted'], ['Fisher', 'unweighted'], ['LargestSite', 'default'], ['Stouffer', 'default'], ['wFisher', 'default'], ['CorrectedStouffer', 'default']]
        lblis = ['Unweighted Stouffer', 'Unweighted Fisher', 'Largest Site', 'Weighted Stouffer', 'wFisher', 'Corrected Stouffer']
        plt.ylim(-0.005, 0.15)
    elif share > 1:
        methLis = [['Stouffer', 'unweighted'], ['Fisher', 'unweighted'], ['LargestSite', 'unweighted'], ['Pearson', 'unweighted'], ['Tippett', 'unweighted']]
        lblis = ['Stouffer', 'Fisher', 'Largest Site', 'Pearson', 'Tippett']
        plt.ylim(-0.005, 0.4)
    for m_idx, meth in enumerate(methLis):
        recArr = np.array(pd.read_csv(os.path.join(dir_name, 's'+str(share)+meth[0]+'_'+meth[1]+'.csv'), index_col=None, header=None))
        pw = recArr.mean(axis = 0)
        std_pw = recArr.std(axis = 0)
        plt.plot(kLis, pw, label = lblis[m_idx], alpha = 0.7, linewidth=4)
        plt.fill_between(kLis, pw-std_pw, pw+std_pw, alpha = 0.3)
    
    plt.plot([kLis[0], kLis[-1]], [0.05, 0.05], alpha = 0.5, color = 'k', label = r"$\alpha$'=0.05", linewidth=3.5)
    desired_order_less_than_one = ['Unweighted Stouffer', 'Unweighted Fisher', 'Weighted Stouffer', 'wFisher', 'Corrected Stouffer', 'Largest Site', r"$\alpha$'=0.05"]
    desired_order_greater_than_one = ['Stouffer', 'Fisher', 'Pearson', 'Tippett', 'Largest Site', r"$\alpha$'=0.05"]

    handles, labels = plt.gca().get_legend_handles_labels()
    if share < 1:
        desired_order = desired_order_less_than_one
        ordered_handles = [h for l, h in sorted(zip(labels, handles), key=lambda lh: desired_order.index(lh[0]))]
        plt.legend(ordered_handles, desired_order, fontsize=35)
    else:
        desired_order = desired_order_greater_than_one
        ordered_handles = [h for l, h in sorted(zip(labels, handles), key=lambda lh: desired_order.index(lh[0]))]
        plt.legend(ordered_handles, desired_order, fontsize=40)
    plt.xlabel('Total Counts', fontsize=40)
    plt.ylabel("Calibrated Confidence Level " + r"$\alpha$'", fontsize=40)
    plt.grid()
    plt.xticks(fontsize=35)
    plt.yticks(fontsize=35)
    plt.tick_params(direction='out', length=10, width=4)
    plt.savefig('s'+str(share)+'.png')
    