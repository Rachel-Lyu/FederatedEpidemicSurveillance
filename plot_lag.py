import matplotlib.pyplot as plt
from bokeh.palettes import Category10
import numpy as np
import pandas as pd
import matplotlib.lines as mlines
import os 
col_set = Category10[10]
dir_name = 'rewrite_lag_hospitalization'
# meth = ['CorrectedStouffer', 'default']
# fig = plt.figure(figsize=(20, 12))
# idx = 0
# for st_idx, [updating_period, training_period, share_lag] in enumerate([[12, 12, 1], [8, 12, 4], [8, 12, 1], [4, 12, 1]]):
#     recallDf = pd.read_csv(os.path.join(dir_name, 'updating'+str(updating_period)+'training'+str(training_period)+'lag'+str(share_lag)+meth[0]+'_'+meth[1]+'_recall.csv'), index_col=None, header=None)
#     precisionDf = pd.read_csv(os.path.join(dir_name, 'updating'+str(updating_period)+'training'+str(training_period)+'lag'+str(share_lag)+meth[0]+'_'+meth[1]+'_precision.csv'), index_col=None, header=None)
#     rr = np.nanmean(np.array(recallDf), axis = 0)
#     pp = np.nanmean(np.array(precisionDf), axis = 0)
#     rp = rr + pp
#     optimal_idx = np.where(rp == np.max(rp))[0][0]
#     tmpx = np.array([0] + [r for r in rr] + [1]).flatten()
#     tmpy = np.array([1] + [p for p in pp] + [0]).flatten()
#     plt.plot(tmpx, tmpy, alpha = 0.5, color = col_set[st_idx], label = 'Upd. '+str(updating_period)+'w, Lag '+str(share_lag)+'w', linestyle = '-', linewidth = 8)
#     plt.plot(rr[optimal_idx], pp[optimal_idx], 'o', color = col_set[st_idx], alpha = 0.5, markersize = 5)
# meth = ['Stouffer', 'unweighted']
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
# plt.legend(fontsize = 20)
# plt.xlabel('Recall', fontsize = 25)
# plt.ylabel('Precision', fontsize = 25)
# plt.xlim(0.7, 1.001)
# plt.ylim(0.7, 1.001)
# plt.savefig('StoufferCorrected_lag_hos.png')

# meth = ['Stouffer', 'default']
# fig = plt.figure(figsize=(20, 12))
# idx = 0
# for st_idx, [updating_period, training_period, share_lag] in enumerate([[12, 12, 1], [8, 12, 4], [8, 12, 1], [4, 12, 1]]):
#     recallDf = pd.read_csv(os.path.join(dir_name, 'updating'+str(updating_period)+'training'+str(training_period)+'lag'+str(share_lag)+meth[0]+'_'+meth[1]+'_recall.csv'), index_col=None, header=None)
#     precisionDf = pd.read_csv(os.path.join(dir_name, 'updating'+str(updating_period)+'training'+str(training_period)+'lag'+str(share_lag)+meth[0]+'_'+meth[1]+'_precision.csv'), index_col=None, header=None)
#     rr = np.nanmean(np.array(recallDf), axis = 0)
#     pp = np.nanmean(np.array(precisionDf), axis = 0)
#     rp = rr + pp
#     optimal_idx = np.where(rp == np.max(rp))[0][0]
#     tmpx = np.array([0] + [r for r in rr] + [1]).flatten()
#     tmpy = np.array([1] + [p for p in pp] + [0]).flatten()
#     plt.plot(tmpx, tmpy, alpha = 0.5, color = col_set[st_idx], label = 'Upd. '+str(updating_period)+'w, Lag '+str(share_lag)+'w', linestyle = '-', linewidth = 8)
#     plt.plot(rr[optimal_idx], pp[optimal_idx], 'o', color = col_set[st_idx], alpha = 0.5, markersize = 5)
# meth = ['Stouffer', 'unweighted']
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
# plt.legend(fontsize = 20)
# plt.xlabel('Recall', fontsize = 25)
# plt.ylabel('Precision', fontsize = 25)
# plt.xlim(0.7, 1.001)
# plt.ylim(0.7, 1.001)
# plt.savefig('StoufferWeighted_lag_hos.png')

# meth = ['Fisher', 'default']
# fig = plt.figure(figsize=(20, 12))
# idx = 0
# for st_idx, [updating_period, training_period, share_lag] in enumerate([[12, 12, 1], [8, 12, 4], [8, 12, 1], [4, 12, 1]]):
#     recallDf = pd.read_csv(os.path.join(dir_name, 'updating'+str(updating_period)+'training'+str(training_period)+'lag'+str(share_lag)+meth[0]+'_'+meth[1]+'_recall.csv'), index_col=None, header=None)
#     precisionDf = pd.read_csv(os.path.join(dir_name, 'updating'+str(updating_period)+'training'+str(training_period)+'lag'+str(share_lag)+meth[0]+'_'+meth[1]+'_precision.csv'), index_col=None, header=None)
#     rr = np.nanmean(np.array(recallDf), axis = 0)
#     pp = np.nanmean(np.array(precisionDf), axis = 0)
#     rp = rr + pp
#     optimal_idx = np.where(rp == np.max(rp))[0][0]
#     tmpx = np.array([0] + [r for r in rr] + [1]).flatten()
#     tmpy = np.array([1] + [p for p in pp] + [0]).flatten()
#     plt.plot(tmpx, tmpy, alpha = 0.5, color = col_set[st_idx], label = 'Upd. '+str(updating_period)+'w, Lag '+str(share_lag)+'w', linestyle = '-', linewidth = 8)
#     plt.plot(rr[optimal_idx], pp[optimal_idx], 'o', color = col_set[st_idx], alpha = 0.5, markersize = 5)
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
# plt.legend(fontsize = 20)
# plt.xlabel('Recall', fontsize = 25)
# plt.ylabel('Precision', fontsize = 25)
# plt.xlim(0.7, 1.001)
# plt.ylim(0.7, 1.001)
# plt.savefig('FisherWeighted_lag_hos.png')

meth = ['wFisher', 'default']
fig = plt.figure(figsize=(20, 12))
idx = 0
for st_idx, [updating_period, training_period, share_lag] in enumerate([[12, 12, 1], [8, 12, 4], [8, 12, 1], [4, 12, 1]]):
    recallDf = pd.read_csv(os.path.join(dir_name, 'updating'+str(updating_period)+'training'+str(training_period)+'lag'+str(share_lag)+meth[0]+'_'+meth[1]+'_recall.csv'), index_col=None, header=None)
    precisionDf = pd.read_csv(os.path.join(dir_name, 'updating'+str(updating_period)+'training'+str(training_period)+'lag'+str(share_lag)+meth[0]+'_'+meth[1]+'_precision.csv'), index_col=None, header=None)
    rr = np.nanmean(np.array(recallDf), axis = 0)
    pp = np.nanmean(np.array(precisionDf), axis = 0)
    rp = rr + pp
    optimal_idx = np.where(rp == np.max(rp))[0][0]
    tmpx = np.array([0] + [r for r in rr] + [1]).flatten()
    tmpy = np.array([1] + [p for p in pp] + [0]).flatten()
    plt.plot(tmpx, tmpy, alpha = 0.5, color = col_set[st_idx], label = 'Upd. '+str(updating_period)+'w, Lag '+str(share_lag)+'w', linestyle = '-', linewidth = 8)
    # plt.plot(rr[optimal_idx], pp[optimal_idx], 'o', color = col_set[st_idx], alpha = 0.5, markersize = 5)
meth = ['Fisher', 'unweighted']
st_idx += 1
recallDf = pd.read_csv(os.path.join(dir_name, 'updating'+str(updating_period)+'training'+str(training_period)+'lag'+str(share_lag)+meth[0]+'_'+meth[1]+'_recall.csv'), index_col=None, header=None)
precisionDf = pd.read_csv(os.path.join(dir_name, 'updating'+str(updating_period)+'training'+str(training_period)+'lag'+str(share_lag)+meth[0]+'_'+meth[1]+'_precision.csv'), index_col=None, header=None)
rr = np.nanmean(np.array(recallDf), axis = 0)
pp = np.nanmean(np.array(precisionDf), axis = 0)
rp = rr + pp
optimal_idx = np.where(rp == np.max(rp))[0][0]
tmpx = np.array([0] + [r for r in rr] + [1]).flatten()
tmpy = np.array([1] + [p for p in pp] + [0]).flatten()
plt.plot(tmpx, tmpy, alpha = 0.5, color = 'k', label = 'Unweighted Fisher', linestyle = ':', linewidth = 10)
# plt.plot(rr[optimal_idx], pp[optimal_idx], 'o', color = 'k', alpha = 0.5, markersize = 5)
plt.legend(fontsize = 35)
plt.xlabel('Recall', fontsize = 40)
plt.ylabel('Precision', fontsize = 40)
plt.xlim(0.7, 1.001)
plt.ylim(0.7, 1.001)
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.tick_params(direction='out', length=10, width=4)
plt.savefig('wFisher_lag_hos.png')

meth = ['CorrectedStouffer', 'default']
fig = plt.figure(figsize=(20, 12))
idx = 0
for st_idx, [updating_period, training_period, share_lag] in enumerate([[12, 12, 1], [8, 12, 4], [8, 12, 1], [4, 12, 1]]):
    recallDf = pd.read_csv(os.path.join(dir_name, 'updating'+str(updating_period)+'training'+str(training_period)+'lag'+str(share_lag)+meth[0]+'_'+meth[1]+'_recall.csv'), index_col=None, header=None)
    precisionDf = pd.read_csv(os.path.join(dir_name, 'updating'+str(updating_period)+'training'+str(training_period)+'lag'+str(share_lag)+meth[0]+'_'+meth[1]+'_precision.csv'), index_col=None, header=None)
    rr = np.nanmean(np.array(recallDf), axis = 0)
    pp = np.nanmean(np.array(precisionDf), axis = 0)
    rp = rr + pp
    optimal_idx = np.where(rp == np.max(rp))[0][0]
    tmpx = np.array([0] + [r for r in rr] + [1]).flatten()
    tmpy = np.array([1] + [p for p in pp] + [0]).flatten()
    plt.plot(tmpx, tmpy, alpha = 0.5, color = col_set[st_idx], label = 'Cont. Corr\'d: Upd. '+str(updating_period)+'w, Lag '+str(share_lag)+'w', linestyle = '-', linewidth = 8)
    # plt.plot(rr[optimal_idx], pp[optimal_idx], 'o', color = col_set[st_idx], alpha = 0.5, markersize = 5)
meth = ['Stouffer', 'default']
idx = 0
for st_idx, [updating_period, training_period, share_lag] in enumerate([[12, 12, 1], [8, 12, 4], [8, 12, 1], [4, 12, 1]]):
    recallDf = pd.read_csv(os.path.join(dir_name, 'updating'+str(updating_period)+'training'+str(training_period)+'lag'+str(share_lag)+meth[0]+'_'+meth[1]+'_recall.csv'), index_col=None, header=None)
    precisionDf = pd.read_csv(os.path.join(dir_name, 'updating'+str(updating_period)+'training'+str(training_period)+'lag'+str(share_lag)+meth[0]+'_'+meth[1]+'_precision.csv'), index_col=None, header=None)
    rr = np.nanmean(np.array(recallDf), axis = 0)
    pp = np.nanmean(np.array(precisionDf), axis = 0)
    rp = rr + pp
    optimal_idx = np.where(rp == np.max(rp))[0][0]
    tmpx = np.array([0] + [r for r in rr] + [1]).flatten()
    tmpy = np.array([1] + [p for p in pp] + [0]).flatten()
    plt.plot(tmpx, tmpy, alpha = 0.5, color = col_set[st_idx], label = 'weighted: Upd. '+str(updating_period)+'w, Lag '+str(share_lag)+'w', linestyle = '--', linewidth = 8)
    # plt.plot(rr[optimal_idx], pp[optimal_idx], 'o', color = col_set[st_idx], alpha = 0.5, markersize = 5)
meth = ['Stouffer', 'unweighted']
recallDf = pd.read_csv(os.path.join(dir_name, 'updating'+str(updating_period)+'training'+str(training_period)+'lag'+str(share_lag)+meth[0]+'_'+meth[1]+'_recall.csv'), index_col=None, header=None)
precisionDf = pd.read_csv(os.path.join(dir_name, 'updating'+str(updating_period)+'training'+str(training_period)+'lag'+str(share_lag)+meth[0]+'_'+meth[1]+'_precision.csv'), index_col=None, header=None)
rr = np.nanmean(np.array(recallDf), axis = 0)
pp = np.nanmean(np.array(precisionDf), axis = 0)
rp = rr + pp
optimal_idx = np.where(rp == np.max(rp))[0][0]
tmpx = np.array([0] + [r for r in rr] + [1]).flatten()
tmpy = np.array([1] + [p for p in pp] + [0]).flatten()
plt.plot(tmpx, tmpy, alpha = 0.5, color = 'k', label = 'Unweighted Stouffer', linestyle = ':', linewidth = 10)
# plt.plot(rr[optimal_idx], pp[optimal_idx], 'o', color = 'k', alpha = 0.5, markersize = 5)
plt.legend(fontsize = 35)
plt.xlabel('Recall', fontsize = 40)
plt.ylabel('Precision', fontsize = 40)
plt.xlim(0.7, 1.001)
plt.ylim(0.7, 1.001)
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.tick_params(direction='out', length=10, width=4)
plt.savefig('Stouffer_lag_hos.png')

dir_name = 'rewrite_lag_claim'
# meth = ['CorrectedStouffer', 'default']
# fig = plt.figure(figsize=(20, 12))
# idx = 0
# for st_idx, [updating_period, training_period, share_lag] in enumerate([[12, 12, 1], [8, 12, 4], [8, 12, 1], [4, 12, 1]]):
#     recallDf = pd.read_csv(os.path.join(dir_name, 'updating'+str(updating_period)+'training'+str(training_period)+'lag'+str(share_lag)+meth[0]+'_'+meth[1]+'_recall.csv'), index_col=None, header=None)
#     precisionDf = pd.read_csv(os.path.join(dir_name, 'updating'+str(updating_period)+'training'+str(training_period)+'lag'+str(share_lag)+meth[0]+'_'+meth[1]+'_precision.csv'), index_col=None, header=None)
#     rr = np.nanmean(np.array(recallDf), axis = 0)
#     pp = np.nanmean(np.array(precisionDf), axis = 0)
#     rp = rr + pp
#     optimal_idx = np.where(rp == np.max(rp))[0][0]
#     tmpx = np.array([0] + [r for r in rr] + [1]).flatten()
#     tmpy = np.array([1] + [p for p in pp] + [0]).flatten()
#     plt.plot(tmpx, tmpy, alpha = 0.5, color = col_set[st_idx], label = 'Upd. '+str(updating_period)+'w, Lag '+str(share_lag)+'w', linestyle = '-', linewidth = 8)
#     plt.plot(rr[optimal_idx], pp[optimal_idx], 'o', color = col_set[st_idx], alpha = 0.5, markersize = 5)
# meth = ['Stouffer', 'unweighted']
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
# plt.legend(fontsize = 20)
# plt.xlabel('Recall', fontsize = 25)
# plt.ylabel('Precision', fontsize = 25)
# plt.xlim(0.7, 1.001)
# plt.ylim(0.7, 1.001)
# plt.savefig('StoufferCorrected_lag_claim.png')

# meth = ['Stouffer', 'default']
# fig = plt.figure(figsize=(20, 12))
# idx = 0
# for st_idx, [updating_period, training_period, share_lag] in enumerate([[12, 12, 1], [8, 12, 4], [8, 12, 1], [4, 12, 1]]):
#     recallDf = pd.read_csv(os.path.join(dir_name, 'updating'+str(updating_period)+'training'+str(training_period)+'lag'+str(share_lag)+meth[0]+'_'+meth[1]+'_recall.csv'), index_col=None, header=None)
#     precisionDf = pd.read_csv(os.path.join(dir_name, 'updating'+str(updating_period)+'training'+str(training_period)+'lag'+str(share_lag)+meth[0]+'_'+meth[1]+'_precision.csv'), index_col=None, header=None)
#     rr = np.nanmean(np.array(recallDf), axis = 0)
#     pp = np.nanmean(np.array(precisionDf), axis = 0)
#     rp = rr + pp
#     optimal_idx = np.where(rp == np.max(rp))[0][0]
#     tmpx = np.array([0] + [r for r in rr] + [1]).flatten()
#     tmpy = np.array([1] + [p for p in pp] + [0]).flatten()
#     plt.plot(tmpx, tmpy, alpha = 0.5, color = col_set[st_idx], label = 'Upd. '+str(updating_period)+'w, Lag '+str(share_lag)+'w', linestyle = '-', linewidth = 8)
#     plt.plot(rr[optimal_idx], pp[optimal_idx], 'o', color = col_set[st_idx], alpha = 0.5, markersize = 5)
# meth = ['Stouffer', 'unweighted']
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
# plt.legend(fontsize = 20)
# plt.xlabel('Recall', fontsize = 25)
# plt.ylabel('Precision', fontsize = 25)
# plt.xlim(0.7, 1.001)
# plt.ylim(0.7, 1.001)
# plt.savefig('StoufferWeighted_lag_claim.png')

# meth = ['Fisher', 'default']
# fig = plt.figure(figsize=(20, 12))
# idx = 0
# for st_idx, [updating_period, training_period, share_lag] in enumerate([[12, 12, 1], [8, 12, 4], [8, 12, 1], [4, 12, 1]]):
#     recallDf = pd.read_csv(os.path.join(dir_name, 'updating'+str(updating_period)+'training'+str(training_period)+'lag'+str(share_lag)+meth[0]+'_'+meth[1]+'_recall.csv'), index_col=None, header=None)
#     precisionDf = pd.read_csv(os.path.join(dir_name, 'updating'+str(updating_period)+'training'+str(training_period)+'lag'+str(share_lag)+meth[0]+'_'+meth[1]+'_precision.csv'), index_col=None, header=None)
#     rr = np.nanmean(np.array(recallDf), axis = 0)
#     pp = np.nanmean(np.array(precisionDf), axis = 0)
#     rp = rr + pp
#     optimal_idx = np.where(rp == np.max(rp))[0][0]
#     tmpx = np.array([0] + [r for r in rr] + [1]).flatten()
#     tmpy = np.array([1] + [p for p in pp] + [0]).flatten()
#     plt.plot(tmpx, tmpy, alpha = 0.5, color = col_set[st_idx], label = 'Upd. '+str(updating_period)+'w, Lag '+str(share_lag)+'w', linestyle = '-', linewidth = 8)
#     plt.plot(rr[optimal_idx], pp[optimal_idx], 'o', color = col_set[st_idx], alpha = 0.5, markersize = 5)
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
# plt.legend(fontsize = 20)
# plt.xlabel('Recall', fontsize = 25)
# plt.ylabel('Precision', fontsize = 25)
# plt.xlim(0.7, 1.001)
# plt.ylim(0.7, 1.001)
# plt.savefig('FisherWeighted_lag_claim.png')

meth = ['wFisher', 'default']
fig = plt.figure(figsize=(20, 12))
idx = 0
for st_idx, [updating_period, training_period, share_lag] in enumerate([[12, 12, 1], [8, 12, 4], [8, 12, 1], [4, 12, 1]]):
    recallDf = pd.read_csv(os.path.join(dir_name, 'updating'+str(updating_period)+'training'+str(training_period)+'lag'+str(share_lag)+meth[0]+'_'+meth[1]+'_recall.csv'), index_col=None, header=None)
    precisionDf = pd.read_csv(os.path.join(dir_name, 'updating'+str(updating_period)+'training'+str(training_period)+'lag'+str(share_lag)+meth[0]+'_'+meth[1]+'_precision.csv'), index_col=None, header=None)
    rr = np.nanmean(np.array(recallDf), axis = 0)
    pp = np.nanmean(np.array(precisionDf), axis = 0)
    rp = rr + pp
    optimal_idx = np.where(rp == np.max(rp))[0][0]
    tmpx = np.array([0] + [r for r in rr] + [1]).flatten()
    tmpy = np.array([1] + [p for p in pp] + [0]).flatten()
    plt.plot(tmpx, tmpy, alpha = 0.5, color = col_set[st_idx], label = 'Upd. '+str(updating_period)+'w, Lag '+str(share_lag)+'w', linestyle = '-', linewidth = 8)
    # plt.plot(rr[optimal_idx], pp[optimal_idx], 'o', color = col_set[st_idx], alpha = 0.5, markersize = 5)
meth = ['Fisher', 'unweighted']
st_idx += 1
recallDf = pd.read_csv(os.path.join(dir_name, 'updating'+str(updating_period)+'training'+str(training_period)+'lag'+str(share_lag)+meth[0]+'_'+meth[1]+'_recall.csv'), index_col=None, header=None)
precisionDf = pd.read_csv(os.path.join(dir_name, 'updating'+str(updating_period)+'training'+str(training_period)+'lag'+str(share_lag)+meth[0]+'_'+meth[1]+'_precision.csv'), index_col=None, header=None)
rr = np.nanmean(np.array(recallDf), axis = 0)
pp = np.nanmean(np.array(precisionDf), axis = 0)
rp = rr + pp
optimal_idx = np.where(rp == np.max(rp))[0][0]
tmpx = np.array([0] + [r for r in rr] + [1]).flatten()
tmpy = np.array([1] + [p for p in pp] + [0]).flatten()
plt.plot(tmpx, tmpy, alpha = 0.5, color = 'k', label = 'Unweighted Fisher', linestyle = ':', linewidth = 10)
# plt.plot(rr[optimal_idx], pp[optimal_idx], 'o', color = 'k', alpha = 0.5, markersize = 5)
plt.legend(fontsize = 35)
plt.xlabel('Recall', fontsize = 40)
plt.ylabel('Precision', fontsize = 40)
plt.xlim(0.7, 1.001)
plt.ylim(0.7, 1.001)
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.tick_params(direction='out', length=10, width=4)
plt.savefig('wFisher_lag_claim.png')

meth = ['CorrectedStouffer', 'default']
fig = plt.figure(figsize=(20, 12))
idx = 0
for st_idx, [updating_period, training_period, share_lag] in enumerate([[12, 12, 1], [8, 12, 4], [8, 12, 1], [4, 12, 1]]):
    recallDf = pd.read_csv(os.path.join(dir_name, 'updating'+str(updating_period)+'training'+str(training_period)+'lag'+str(share_lag)+meth[0]+'_'+meth[1]+'_recall.csv'), index_col=None, header=None)
    precisionDf = pd.read_csv(os.path.join(dir_name, 'updating'+str(updating_period)+'training'+str(training_period)+'lag'+str(share_lag)+meth[0]+'_'+meth[1]+'_precision.csv'), index_col=None, header=None)
    rr = np.nanmean(np.array(recallDf), axis = 0)
    pp = np.nanmean(np.array(precisionDf), axis = 0)
    rp = rr + pp
    optimal_idx = np.where(rp == np.max(rp))[0][0]
    tmpx = np.array([0] + [r for r in rr] + [1]).flatten()
    tmpy = np.array([1] + [p for p in pp] + [0]).flatten()
    plt.plot(tmpx, tmpy, alpha = 0.5, color = col_set[st_idx], label = 'Cont. Corr\'d: Upd. '+str(updating_period)+'w, Lag '+str(share_lag)+'w', linestyle = '-', linewidth = 8)
    # plt.plot(rr[optimal_idx], pp[optimal_idx], 'o', color = col_set[st_idx], alpha = 0.5, markersize = 5)
meth = ['Stouffer', 'default']
idx = 0
for st_idx, [updating_period, training_period, share_lag] in enumerate([[12, 12, 1], [8, 12, 4], [8, 12, 1], [4, 12, 1]]):
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
meth = ['Stouffer', 'unweighted']
recallDf = pd.read_csv(os.path.join(dir_name, 'updating'+str(updating_period)+'training'+str(training_period)+'lag'+str(share_lag)+meth[0]+'_'+meth[1]+'_recall.csv'), index_col=None, header=None)
precisionDf = pd.read_csv(os.path.join(dir_name, 'updating'+str(updating_period)+'training'+str(training_period)+'lag'+str(share_lag)+meth[0]+'_'+meth[1]+'_precision.csv'), index_col=None, header=None)
rr = np.nanmean(np.array(recallDf), axis = 0)
pp = np.nanmean(np.array(precisionDf), axis = 0)
rp = rr + pp
optimal_idx = np.where(rp == np.max(rp))[0][0]
tmpx = np.array([0] + [r for r in rr] + [1]).flatten()
tmpy = np.array([1] + [p for p in pp] + [0]).flatten()
plt.plot(tmpx, tmpy, alpha = 0.5, color = 'k', label = 'Unweighted Stouffer', linestyle = ':', linewidth = 10)
# plt.plot(rr[optimal_idx], pp[optimal_idx], 'o', color = 'k', alpha = 0.5, markersize = 5)


handle_solid = mlines.Line2D([], [], color='k', alpha = 0.5, linestyle='-', linewidth=7, label='Cont. Corr\'d')
handle_dashed = mlines.Line2D([], [], color='k', alpha = 0.5, linestyle = '--', linewidth = 8, label='Weighted')
legend_handles = [handle_solid, handle_dashed]
for st_idx, [updating_period, training_period, share_lag] in enumerate([[12, 12, 1], [8, 12, 4], [8, 12, 1], [4, 12, 1]]):
    legend_handles.append(mlines.Line2D([], [], alpha = 0.5, color = col_set[st_idx], linewidth=7, label = 'Upd. '+str(updating_period)+'w, Lag '+str(share_lag)+'w'))
legend_handles.append(mlines.Line2D([], [], alpha = 0.5, color = 'k', label = 'Unweighted Stouffer', linestyle = ':', linewidth = 10))
# print(legend_handles)
plt.legend(handles=legend_handles, fontsize = 35)
plt.xlabel('Recall', fontsize = 40)
plt.ylabel('Precision', fontsize = 40)
plt.xlim(0.65, 1.001)
plt.ylim(0.65, 1.001)
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.tick_params(direction='out', length=10, width=4)
plt.savefig('Stouffer_lag_claim.png')