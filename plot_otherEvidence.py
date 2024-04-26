import matplotlib.pyplot as plt
from bokeh.palettes import Category10
import numpy as np
import pandas as pd
import os 
col_set = [Category10[10][0], Category10[10][3], Category10[10][1], Category10[10][2]]
dir_name = 'rewrite_lag_hospitalization'
allocLis = ['inpatient_beds_used_7_day_sum', 'total_icu_beds_7_day_sum']
auxLis = ['Weighted: Inpatient Beds', 'Weighted: ICU Beds']
# meth = ['wFisher', 'default']
# fig = plt.figure(figsize=(10, 6))
# idx = 0
# for st_idx, [updating_period, training_period, share_lag] in enumerate([[12, 12, 1], [4, 12, 1]]):
#     recallDf = pd.read_csv(os.path.join(dir_name, 'updating'+str(updating_period)+'training'+str(training_period)+'lag'+str(share_lag)+meth[0]+'_'+meth[1]+'_recall.csv'), index_col=None, header=None)
#     precisionDf = pd.read_csv(os.path.join(dir_name, 'updating'+str(updating_period)+'training'+str(training_period)+'lag'+str(share_lag)+meth[0]+'_'+meth[1]+'_precision.csv'), index_col=None, header=None)
#     rr = np.nanmean(np.array(recallDf), axis = 0)
#     pp = np.nanmean(np.array(precisionDf), axis = 0)
#     rp = rr + pp
#     optimal_idx = np.where(rp == np.max(rp))[0][0]
#     tmpx = np.array([0] + [r for r in rr] + [1]).flatten()
#     tmpy = np.array([1] + [p for p in pp] + [0]).flatten()
#     plt.plot(tmpx, tmpy, alpha = 0.5, color = col_set[st_idx], label = 'updating'+str(updating_period)+'w, lag'+str(share_lag)+'w', linestyle = '-', linewidth = 8)
#     plt.plot(rr[optimal_idx], pp[optimal_idx], 'o', color = col_set[st_idx], alpha = 0.5, markersize = 5)
# for mt_idx, allocMeth in enumerate(allocLis):
#     recallDf = pd.read_csv(os.path.join(dir_name, allocMeth+'_'+meth[0]+'_'+meth[1]+'_recall.csv'), index_col=None, header=None)
#     precisionDf = pd.read_csv(os.path.join(dir_name, allocMeth+'_'+meth[0]+'_'+meth[1]+'_precision.csv'), index_col=None, header=None)
#     rr = np.nanmean(np.array(recallDf), axis = 0)
#     pp = np.nanmean(np.array(precisionDf), axis = 0)
#     rp = rr + pp
#     optimal_idx = np.where(rp == np.max(rp))[0][0]
#     tmpx = np.array([0] + [r for r in rr] + [1]).flatten()
#     tmpy = np.array([1] + [p for p in pp] + [0]).flatten()
#     plt.plot(tmpx, tmpy, alpha = 0.5, color = col_set[mt_idx+st_idx+1], label = auxLis[mt_idx], linestyle = '-', linewidth = 8)
#     plt.plot(rr[optimal_idx], pp[optimal_idx], 'o', color = col_set[mt_idx+st_idx+1], alpha = 0.5, markersize = 5)
# meth = ['Fisher', 'unweighted']
# st_idx += 1
# recallDf = pd.read_csv(os.path.join(dir_name, 'updating'+str(updating_period)+'training'+str(training_period)+'lag'+str(share_lag)+meth[0]+'_'+meth[1]+'_recall.csv'), index_col=None, header=None)
# precisionDf = pd.read_csv(os.path.join(dir_name, 'updating'+str(updating_period)+'training'+str(training_period)+'lag'+str(share_lag)+meth[0]+'_'+meth[1]+'_precision.csv'), index_col=None, header=None)
# rr = np.nanmean(np.array(recallDf), axis = 0)
# pp = np.nanmean(np.array(precisionDf), axis = 0)
# rp = rr + pp
# optimal_idx = np.where(rp == np.max(rp))[0][0]
# tmpx = np.array([0] + [r for r in rr] + [1]).flatten()
# tmpy = np.array([1] + [p for p in pp] + [0]).flatten()
# plt.plot(tmpx, tmpy, alpha = 0.5, color = 'k', label = 'unweighted', linestyle = '-', linewidth = 8)
# plt.plot(rr[optimal_idx], pp[optimal_idx], 'o', color = 'k', alpha = 0.5, markersize = 5)
# plt.legend(fontsize = 12)
# plt.xlabel('Recall', fontsize = 15)
# plt.ylabel('Precision', fontsize = 15)
# plt.xlim(0.7, 1.001)
# plt.ylim(0.7, 1.001)
# plt.savefig('wFisher_lag_otherEvidence_hos.png')

# meth = ['CorrectedStouffer', 'default']
fig = plt.figure(figsize=(20, 12))
# idx = 0
# for st_idx, [updating_period, training_period, share_lag] in enumerate([[12, 12, 1], [4, 12, 1]]):
#     recallDf = pd.read_csv(os.path.join(dir_name, 'updating'+str(updating_period)+'training'+str(training_period)+'lag'+str(share_lag)+meth[0]+'_'+meth[1]+'_recall.csv'), index_col=None, header=None)
#     precisionDf = pd.read_csv(os.path.join(dir_name, 'updating'+str(updating_period)+'training'+str(training_period)+'lag'+str(share_lag)+meth[0]+'_'+meth[1]+'_precision.csv'), index_col=None, header=None)
#     rr = np.nanmean(np.array(recallDf), axis = 0)
#     pp = np.nanmean(np.array(precisionDf), axis = 0)
#     rp = rr + pp
#     optimal_idx = np.where(rp == np.max(rp))[0][0]
#     tmpx = np.array([0] + [r for r in rr] + [1]).flatten()
#     tmpy = np.array([1] + [p for p in pp] + [0]).flatten()
#     plt.plot(tmpx, tmpy, alpha = 0.5, color = col_set[st_idx], label = 'corrected: updating'+str(updating_period)+'w, lag'+str(share_lag)+'w', linestyle = '-', linewidth = 8)
#     plt.plot(rr[optimal_idx], pp[optimal_idx], 'o', color = col_set[st_idx], alpha = 0.5, markersize = 5)
meth = ['Stouffer', 'default']
idx = 0
for st_idx, [updating_period, training_period, share_lag] in enumerate([[12, 12, 1], [4, 12, 1]]):
    recallDf = pd.read_csv(os.path.join(dir_name, 'updating'+str(updating_period)+'training'+str(training_period)+'lag'+str(share_lag)+meth[0]+'_'+meth[1]+'_recall.csv'), index_col=None, header=None)
    precisionDf = pd.read_csv(os.path.join(dir_name, 'updating'+str(updating_period)+'training'+str(training_period)+'lag'+str(share_lag)+meth[0]+'_'+meth[1]+'_precision.csv'), index_col=None, header=None)
    rr = np.nanmean(np.array(recallDf), axis = 0)
    pp = np.nanmean(np.array(precisionDf), axis = 0)
    rp = rr + pp
    optimal_idx = np.where(rp == np.max(rp))[0][0]
    tmpx = np.array([0] + [r for r in rr] + [1]).flatten()
    tmpy = np.array([1] + [p for p in pp] + [0]).flatten()
    plt.plot(tmpx, tmpy, alpha = 0.5, color = col_set[st_idx], label = 'Weighted: Upd. '+str(updating_period)+'w, Lag '+str(share_lag)+'w', linestyle = '--', linewidth = 8)
    # plt.plot(rr[optimal_idx], pp[optimal_idx], 'o', color = col_set[st_idx], alpha = 0.5, markersize = 5)
for mt_idx, allocMeth in enumerate(allocLis):
    recallDf = pd.read_csv(os.path.join(dir_name, allocMeth+'_'+meth[0]+'_'+meth[1]+'_recall.csv'), index_col=None, header=None)
    precisionDf = pd.read_csv(os.path.join(dir_name, allocMeth+'_'+meth[0]+'_'+meth[1]+'_precision.csv'), index_col=None, header=None)
    rr = np.nanmean(np.array(recallDf), axis = 0)
    pp = np.nanmean(np.array(precisionDf), axis = 0)
    rp = rr + pp
    optimal_idx = np.where(rp == np.max(rp))[0][0]
    tmpx = np.array([0] + [r for r in rr] + [1]).flatten()
    tmpy = np.array([1] + [p for p in pp] + [0]).flatten()
    plt.plot(tmpx, tmpy, alpha = 0.5, color = col_set[mt_idx+st_idx+1], label = auxLis[mt_idx], linestyle = '-', linewidth = 8)
    # plt.plot(rr[optimal_idx], pp[optimal_idx], 'o', color = col_set[mt_idx+st_idx+1], alpha = 0.5, markersize = 5)
meth = ['Stouffer', 'unweighted']
recallDf = pd.read_csv(os.path.join(dir_name, 'updating'+str(updating_period)+'training'+str(training_period)+'lag'+str(share_lag)+meth[0]+'_'+meth[1]+'_recall.csv'), index_col=None, header=None)
precisionDf = pd.read_csv(os.path.join(dir_name, 'updating'+str(updating_period)+'training'+str(training_period)+'lag'+str(share_lag)+meth[0]+'_'+meth[1]+'_precision.csv'), index_col=None, header=None)
rr = np.nanmean(np.array(recallDf), axis = 0)
pp = np.nanmean(np.array(precisionDf), axis = 0)
rp = rr + pp
optimal_idx = np.where(rp == np.max(rp))[0][0]
tmpx = np.array([0] + [r for r in rr] + [1]).flatten()
tmpy = np.array([1] + [p for p in pp] + [0]).flatten()
plt.plot(tmpx, tmpy, alpha = 0.5, color = 'k', label = 'Unweighted Stouffer', linestyle = '-', linewidth = 8)
# plt.plot(rr[optimal_idx], pp[optimal_idx], 'o', color = 'k', alpha = 0.5, markersize = 5)
plt.legend(fontsize = 40)
plt.xlabel('Recall', fontsize = 40)
plt.ylabel('Precision', fontsize = 40)
plt.xlim(0.7, 1.001)
plt.ylim(0.7, 1.001)
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.savefig('Stouffer_lag_otherEvidence_hos.png')
