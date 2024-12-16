import pandas as pd
import numpy as np
from scipy.special import ndtri
import scipy.stats as stats
import matplotlib.pyplot as plt
import scipy.stats as stats
import argparse
import os

k = 200
null_theta = 0.3
new_dir_name = 'power_curve'
real_theta = np.linspace(0, 1.3, 50)
for share in [0.6, 0.8, 0.9, 2.0, 3.0, 5.0, 8.0, 10.0]:
    plt.figure(figsize = (20, 12))
    # if share < 1:
    #     methLis = [['Stouffer', 'unweighted'], ['Fisher', 'unweighted'], ['Stouffer', 'default'], ['Fisher', 'default'], ['wFisher', 'default'], ['Pearson', 'unweighted'], ['Tippett', 'unweighted'], ['LargestSite', 'default']]
    #     lblis = ['Unweighted Stouffer', 'Unweighted Fisher', 'Weighted Stouffer', 'Weighted Fisher', 'wFisher', 'Pearson', 'Tippett', 'Largest Site']
    # elif share > 1:
    #     methLis = [['Stouffer', 'unweighted'], ['Fisher', 'unweighted'], ['Pearson', 'unweighted'], ['Tippett', 'unweighted'], ['LargestSite', 'unweighted']]
    #     lblis = ['Stouffer', 'Fisher', 'Pearson', 'Tippett', 'Largest Site']
    if share < 1:
        methLis = [['Stouffer', 'unweighted'], ['Fisher', 'unweighted'], ['LargestSite', 'default'], ['Stouffer', 'default'], ['wFisher', 'default'], ['CorrectedStouffer', 'default']]
        lblis = ['Unweighted Stouffer', 'Unweighted Fisher', 'Largest Site', 'Weighted Stouffer', 'wFisher', 'Corrected Stouffer']
    elif share > 1:
        methLis = [['Stouffer', 'unweighted'], ['Fisher', 'unweighted'], ['LargestSite', 'unweighted'], ['Pearson', 'unweighted'], ['Tippett', 'unweighted']]
        lblis = ['Stouffer', 'Fisher', 'Largest Site', 'Pearson', 'Tippett']
    for m_idx, meth in enumerate(methLis):
        recArr = np.array(pd.read_csv(os.path.join(new_dir_name, 's'+str(share)+meth[0]+'_'+meth[1]+'_k'+str(k)+'.csv'), index_col=None, header=None))
        pw = recArr.mean(axis = 0)
        std_pw = recArr.std(axis = 0)
        plt.plot(real_theta, pw, label = lblis[m_idx], alpha = 0.7, linewidth=6)
        # plt.fill_between(real_theta, pw-std_pw, pw+std_pw, alpha = 0.3)
    
    recArr = np.array(pd.read_csv(os.path.join(new_dir_name, 'BinomialPower.csv'), index_col=None, header=None)).flatten()
    plt.plot(real_theta, recArr, color = 'k', linestyle = '--', label = 'Binomial Power', alpha = 0.7, linewidth=6)
    plt.plot([null_theta, null_theta], [0, 0.2], alpha = 0.5, color = 'k', linewidth=5)
    plt.plot([null_theta-0.2, null_theta+0.2], [0.05, 0.05], alpha = 0.5, color = 'k', label = '(30%, 0.05)', linewidth=5)
    desired_order_less_than_one = ['Binomial Power', 'Unweighted Stouffer', 'Unweighted Fisher', 'Weighted Stouffer', 'wFisher', 'Corrected Stouffer', 'Largest Site', '(30%, 0.05)']
    desired_order_greater_than_one = ['Binomial Power', 'Stouffer', 'Fisher', 'Pearson', 'Tippett', 'Largest Site', '(30%, 0.05)']

    handles, labels = plt.gca().get_legend_handles_labels()
    if share < 1:
        desired_order = desired_order_less_than_one
        ordered_handles = [h for l, h in sorted(zip(labels, handles), key=lambda lh: desired_order.index(lh[0]))]
        plt.legend(ordered_handles, desired_order, fontsize=35)
    else:
        desired_order = desired_order_greater_than_one
        ordered_handles = [h for l, h in sorted(zip(labels, handles), key=lambda lh: desired_order.index(lh[0]))]
        plt.legend(ordered_handles, desired_order, fontsize=40)

    plt.grid()
    plt.xlabel('Real Growth Rate', fontsize=40)
    plt.ylabel('FPR/Power', fontsize=40)
    plt.xticks(fontsize=35)
    plt.yticks(fontsize=35)
    plt.tick_params(direction='out', length=10, width=4)
    plt.savefig('pw'+str(share)+'.png')
    