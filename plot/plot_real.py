import matplotlib.pyplot as plt
from bokeh.palettes import Category10
import numpy as np
import pandas as pd
import os 
dir_name = 'rewrite_mainRes_claim'
col_set = [Category10[10][0], Category10[10][1], Category10[10][2], Category10[10][3], Category10[10][4]]
cnt_nm = ['TP', 'FN', 'FP']
methLis = [['Stouffer', 'unweighted'], ['Fisher', 'unweighted'], ['LargestSite', 'default'], ['Pearson', 'unweighted'], ['Tippett', 'unweighted']]
# lblis = ['Stouffer\'s Z', 'Fisher\'s log(p)', 'Largest Site', 'Pearson\'s log(1-p)', 'Tippett\'s min(p)']
lblis = ['Stouffer', 'Fisher', 'Largest Site', 'Pearson', 'Tippett']
desired_order = ['Stouffer', 'Fisher', 'Pearson', 'Tippett', 'Largest Site']
fig, ax = plt.subplots(figsize = (20, 12))
idx = 0
for meth_idx, meth in enumerate(methLis):
    recallDf = pd.read_csv(os.path.join(dir_name, meth[0]+'_'+meth[1]+'_recall.csv'), index_col=None, header=None)
    precisionDf = pd.read_csv(os.path.join(dir_name, meth[0]+'_'+meth[1]+'_precision.csv'), index_col=None, header=None)
    rr = np.nanmean(np.array(recallDf), axis = 0)
    pp = np.nanmean(np.array(precisionDf), axis = 0)
    rp = rr + pp
    optimal_idx = np.where(rp == np.max(rp))[0][0]
    tmpx = np.array([0] + [r for r in rr] + [1]).flatten()
    tmpy = np.array([1] + [p for p in pp] + [0]).flatten()
    plt.plot(tmpx, tmpy, alpha = 0.5, color = col_set[meth_idx], label = lblis[meth_idx], linestyle = '-', linewidth = 8)
    # plt.plot(rr[optimal_idx], pp[optimal_idx], 'o', color = col_set[meth_idx], alpha = 0.5, markersize = 5)
handles, labels = plt.gca().get_legend_handles_labels()
ordered_handles = [h for l, h in sorted(zip(labels, handles), key=lambda lh: desired_order.index(lh[0]))]
plt.xlabel('Recall', fontsize=40)
plt.ylabel('Precision', fontsize=40)
plt.xlim(0.6, 1.001)
plt.ylim(0.6, 1.001)
plt.legend(ordered_handles, desired_order, fontsize=40)
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.tick_params(direction='out', length=10, width=4)
plt.savefig('re_unwt_ct.png')

methLis = [['Stouffer', 'unweighted'], ['Fisher', 'unweighted'], ['LargestSite', 'default'], ['Stouffer', 'default'], ['wFisher', 'default'], ['CorrectedStouffer', 'default']]
# lblis = ['Unweighted Stouffer', 'Unweighted Fisher', 'Largest Site', 'Weighted Stouffer', 'wFisher', 'Corrected Stouffer']
lblis = ['Unweighted Stouffer', 'Unweighted Fisher', 'Largest Site', 'Weighted Stouffer', 'wFisher', 'Corrected Stouffer']
desired_order = ['Unweighted Stouffer', 'Unweighted Fisher', 'Weighted Stouffer', 'wFisher', 'Corrected Stouffer', 'Largest Site']
lstLis = ['-', '-', '-', '--', '--', ':']
lwdLis = [7, 7, 7, 8, 8, 9]
col_set = [Category10[10][0], Category10[10][1], Category10[10][2], Category10[10][0], Category10[10][1], Category10[10][0]]
fig, ax = plt.subplots(figsize = (20, 12))
idx = 0
for meth_idx, meth in enumerate(methLis):
    recallDf = pd.read_csv(os.path.join(dir_name, meth[0]+'_'+meth[1]+'_recall.csv'), index_col=None, header=None)
    precisionDf = pd.read_csv(os.path.join(dir_name, meth[0]+'_'+meth[1]+'_precision.csv'), index_col=None, header=None)
    rr = np.nanmean(np.array(recallDf), axis = 0)
    pp = np.nanmean(np.array(precisionDf), axis = 0)
    rp = rr + pp
    optimal_idx = np.where(rp == np.max(rp))[0][0]
    tmpx = np.array([0] + [r for r in rr] + [1]).flatten()
    tmpy = np.array([1] + [p for p in pp] + [0]).flatten()
    plt.plot(tmpx, tmpy, alpha = 0.5, color = col_set[meth_idx], label = lblis[meth_idx], linestyle = lstLis[meth_idx], linewidth = lwdLis[meth_idx])
    # plt.plot(rr[optimal_idx], pp[optimal_idx], 'o', color = col_set[meth_idx], alpha = 0.5, markersize = 5)
handles, labels = plt.gca().get_legend_handles_labels()
ordered_handles = [h for l, h in sorted(zip(labels, handles), key=lambda lh: desired_order.index(lh[0]))]
plt.xlabel('Recall', fontsize=40)
plt.ylabel('Precision', fontsize=40)
plt.xlim(0.6, 1.001)
plt.ylim(0.6, 1.001)
plt.legend(ordered_handles, desired_order, fontsize=40)
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.tick_params(direction='out', length=10, width=4)
plt.savefig('re_wt_ct.png')

dir_name = 'rewrite_mainRes_hospitalization'
col_set = [Category10[10][0], Category10[10][1], Category10[10][2], Category10[10][3], Category10[10][4]]
methLis = [['Stouffer', 'unweighted'], ['Fisher', 'unweighted'], ['LargestSite', 'default'], ['Pearson', 'unweighted'], ['Tippett', 'unweighted']]
# lblis = ['Stouffer\'s Z', 'Fisher\'s log(p)', 'Largest Site', 'Pearson\'s log(1-p)', 'Tippett\'s min(p)']
lblis = ['Stouffer', 'Fisher', 'Largest Site', 'Pearson', 'Tippett']
desired_order = ['Stouffer', 'Fisher', 'Pearson', 'Tippett', 'Largest Site']
fig, ax = plt.subplots(figsize = (20, 12))
idx = 0
for meth_idx, meth in enumerate(methLis):
    recallDf = pd.read_csv(os.path.join(dir_name, meth[0]+'_'+meth[1]+'_recall.csv'), index_col=None, header=None)
    precisionDf = pd.read_csv(os.path.join(dir_name, meth[0]+'_'+meth[1]+'_precision.csv'), index_col=None, header=None)
    rr = np.nanmean(np.array(recallDf), axis = 0)
    pp = np.nanmean(np.array(precisionDf), axis = 0)
    rp = rr + pp
    optimal_idx = np.where(rp == np.max(rp))[0][0]
    tmpx = np.array([0] + [r for r in rr] + [1]).flatten()
    tmpy = np.array([1] + [p for p in pp] + [0]).flatten()
    plt.plot(tmpx, tmpy, alpha = 0.5, color = col_set[meth_idx], label = lblis[meth_idx], linestyle = '-', linewidth = 8)
    # plt.plot(rr[optimal_idx], pp[optimal_idx], 'o', color = col_set[meth_idx], alpha = 0.5, markersize = 5)
handles, labels = plt.gca().get_legend_handles_labels()
ordered_handles = [h for l, h in sorted(zip(labels, handles), key=lambda lh: desired_order.index(lh[0]))]
plt.xlabel('Recall', fontsize=40)
plt.ylabel('Precision', fontsize=40)
plt.xlim(0.6, 1.001)
plt.ylim(0.6, 1.001)
plt.legend(ordered_handles, desired_order, fontsize=40)
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.tick_params(direction='out', length=10, width=4)
plt.savefig('re_unwt_fac.png')

methLis = [['Stouffer', 'unweighted'], ['Fisher', 'unweighted'], ['LargestSite', 'default'], ['Stouffer', 'default'], ['wFisher', 'default'], ['CorrectedStouffer', 'default']]
lblis = ['Unweighted Stouffer', 'Unweighted Fisher', 'Largest Site', 'Weighted Stouffer', 'wFisher', 'Corrected Stouffer']
desired_order = ['Unweighted Stouffer', 'Unweighted Fisher', 'Weighted Stouffer', 'wFisher', 'Corrected Stouffer', 'Largest Site']
lstLis = ['-', '-', '-', '--', '--', ':']
lwdLis = [7, 7, 7, 8, 8, 9]
col_set = [Category10[10][0], Category10[10][1], Category10[10][2], Category10[10][0], Category10[10][1], Category10[10][0]]
fig, ax = plt.subplots(figsize = (20, 12))
idx = 0
for meth_idx, meth in enumerate(methLis):
    recallDf = pd.read_csv(os.path.join(dir_name, meth[0]+'_'+meth[1]+'_recall.csv'), index_col=None, header=None)
    precisionDf = pd.read_csv(os.path.join(dir_name, meth[0]+'_'+meth[1]+'_precision.csv'), index_col=None, header=None)
    rr = np.nanmean(np.array(recallDf), axis = 0)
    pp = np.nanmean(np.array(precisionDf), axis = 0)
    rp = rr + pp
    optimal_idx = np.where(rp == np.max(rp))[0][0]
    tmpx = np.array([0] + [r for r in rr] + [1]).flatten()
    tmpy = np.array([1] + [p for p in pp] + [0]).flatten()
    plt.plot(tmpx, tmpy, alpha = 0.5, color = col_set[meth_idx], label = lblis[meth_idx], linestyle = lstLis[meth_idx], linewidth = lwdLis[meth_idx])
    # plt.plot(rr[optimal_idx], pp[optimal_idx], 'o', color = col_set[meth_idx], alpha = 0.5, markersize = 5)
handles, labels = plt.gca().get_legend_handles_labels()
ordered_handles = [h for l, h in sorted(zip(labels, handles), key=lambda lh: desired_order.index(lh[0]))]
plt.xlabel('Recall', fontsize=40)
plt.ylabel('Precision', fontsize=40)
plt.xlim(0.6, 1.001)
plt.ylim(0.6, 1.001)
plt.legend(ordered_handles, desired_order, fontsize=40)
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.tick_params(direction='out', length=10, width=4)
plt.savefig('re_wt_fac.png')
