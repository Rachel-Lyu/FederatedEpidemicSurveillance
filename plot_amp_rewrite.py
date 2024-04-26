import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bokeh.palettes import Category10
import os

def genMetricsFromFiles(TPfile, FNfile, FPfile):
    TPdf = pd.read_csv(TPfile, index_col=None, header=None)
    FNdf = pd.read_csv(FNfile, index_col=None, header=None)
    FPdf = pd.read_csv(FPfile, index_col=None, header=None)
    
    recall = TPdf/(FNdf+TPdf)
    precision = TPdf/(FPdf+TPdf)
    precision = precision.fillna(1)

    recallMean = np.mean(recall, axis = 0)
    precisionMean = np.mean(precision, axis = 0)

    recallStd = np.std(recall, axis = 0)
    precisionStd = np.std(precision, axis = 0)

    return recallMean, precisionMean, recallStd, precisionStd

dir_name = '~/semiSyn/'

def get_power(TPfile, FNfile, FPfile, FDR):
    TPdf = pd.read_csv(TPfile, index_col=None, header=None)
    FNdf = pd.read_csv(FNfile, index_col=None, header=None)
    FPdf = pd.read_csv(FPfile, index_col=None, header=None)
    
    recall = TPdf/(FNdf+TPdf)
    precision = TPdf/(FPdf+TPdf)
    precision = precision.fillna(1)

    recall = np.mean(recall, axis = 0)
    precision = np.mean(precision, axis = 0)

    ctrl = 1 - FDR
    
    acceptance = np.where(precision >= ctrl)[0]
    if (len(acceptance)>0 and acceptance[-1]<len(precision)-1):
        lptr = acceptance[-1]
        rptr = acceptance[-1]+1
        wt = (precision[lptr] - ctrl)/(precision[lptr] - precision[rptr])
        power = wt * recall[rptr] + (1-wt) * recall[lptr]
        return np.round(power, 6)
    else:
        return None

dir_name = '~/amplification/'
col_set = Category10[10]
cnt_nm = ['TP', 'FN', 'FP']
methLis = [['Stouffer', 'unweighted'], ['Fisher', 'unweighted'], ['LargestSite', 'unweighted'], ['Pearson', 'unweighted'], ['Tippett', 'unweighted']]
# lbLis = [r'Stouffer ($Z$)', r'Fisher ($\log p$)', r'Largest Site', r'Pearson ($\log 1-p$)', r'Tippett $\min p$']
lbLis = ['Stouffer', 'Fisher', 'Largest Site', 'Pearson', 'Tippett']
ampLis = [0.2, 0.25, 0.33, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
numSites = 5
# fnm = 'centralized_withNoise'
# cen_power_list = []
# for idx, amplification in enumerate(ampLis):
#     cen_power_list.append(get_power(os.path.join(dir_name, methLis[0][0], cnt_nm[0]+'_'+fnm+'_share'+str(numSites)+'_amp'+str(amplification)+'.csv'), 
#     os.path.join(dir_name, methLis[0][0], cnt_nm[1]+'_'+fnm+'_share'+str(numSites)+'_amp'+str(amplification)+'.csv'), 
#     os.path.join(dir_name, methLis[0][0], cnt_nm[2]+'_'+fnm+'_share'+str(numSites)+'_amp'+str(amplification)+'.csv'), 0.1))
# cen_power_list = np.array(cen_power_list)
# fig, ax = plt.subplots(figsize = (10, 6))
# plt.plot(ampLis, np.repeat(np.mean(cen_power_list), len(ampLis)), label = 'centralized', alpha = 0.3, color = 'k', linewidth = 2)
# cen_power_list -= np.mean(cen_power_list)
# fnm = 'decentralized_withNoise'
# meth_power_list = []
# for meth_idx, meth in enumerate(methLis):
#     share_power_list = []
#     for idx, amplification in enumerate(ampLis):
#         share_power_list.append(get_power(os.path.join(dir_name, meth[0], cnt_nm[0]+'_'+fnm+'_share'+str(numSites)+'_amp'+str(amplification)+'.csv'), 
#         os.path.join(dir_name, meth[0], cnt_nm[1]+'_'+fnm+'_share'+str(numSites)+'_amp'+str(amplification)+'.csv'), 
#         os.path.join(dir_name, meth[0], cnt_nm[2]+'_'+fnm+'_share'+str(numSites)+'_amp'+str(amplification)+'.csv'), 0.1))
#     meth_power_list.append(share_power_list)

# cnt_dct = {methLis[i][0]+'_'+methLis[i][1]: meth_power_list[i] for i in range(len(methLis))}

# for meth_idx, meth in enumerate(methLis):
#     plt.plot(ampLis, cnt_dct[meth[0]+'_'+meth[1]]-cen_power_list, label = lbLis[meth_idx], alpha = 0.6, color = col_set[meth_idx], linewidth = 2)
#     plt.scatter(ampLis, cnt_dct[meth[0]+'_'+meth[1]]-cen_power_list, alpha = 0.6, color = col_set[meth_idx])

# ax.set_xlabel('number of sites')
# ax.set_ylabel('power')
# plt.ylim(0.8, 1)
# ax.legend()
# plt.savefig('rewrite_power_5_amp.png')

# numSites = 10
# fnm = 'centralized_withNoise'
# cen_power_list = []
# for idx, amplification in enumerate(ampLis):
#     cen_power_list.append(get_power(os.path.join(dir_name, methLis[0][0], cnt_nm[0]+'_'+fnm+'_share'+str(numSites)+'_amp'+str(amplification)+'.csv'), 
#     os.path.join(dir_name, methLis[0][0], cnt_nm[1]+'_'+fnm+'_share'+str(numSites)+'_amp'+str(amplification)+'.csv'), 
#     os.path.join(dir_name, methLis[0][0], cnt_nm[2]+'_'+fnm+'_share'+str(numSites)+'_amp'+str(amplification)+'.csv'), 0.1))
# cen_power_list = np.array(cen_power_list)
# fig, ax = plt.subplots(figsize = (10, 6))
# plt.plot(ampLis, np.repeat(np.mean(cen_power_list), len(ampLis)), label = 'centralized', alpha = 0.3, color = 'k', linewidth = 2)
# cen_power_list -= np.mean(cen_power_list)
# fnm = 'decentralized_withNoise'
# meth_power_list = []
# for meth_idx, meth in enumerate(methLis):
#     share_power_list = []
#     for idx, amplification in enumerate(ampLis):
#         share_power_list.append(get_power(os.path.join(dir_name, meth[0], cnt_nm[0]+'_'+fnm+'_share'+str(numSites)+'_amp'+str(amplification)+'.csv'), 
#         os.path.join(dir_name, meth[0], cnt_nm[1]+'_'+fnm+'_share'+str(numSites)+'_amp'+str(amplification)+'.csv'), 
#         os.path.join(dir_name, meth[0], cnt_nm[2]+'_'+fnm+'_share'+str(numSites)+'_amp'+str(amplification)+'.csv'), 0.1))
#     meth_power_list.append(share_power_list)

# cnt_dct = {methLis[i][0]+'_'+methLis[i][1]: meth_power_list[i] for i in range(len(methLis))}

# for meth_idx, meth in enumerate(methLis):
#     plt.plot(ampLis, cnt_dct[meth[0]+'_'+meth[1]]-cen_power_list, label = lbLis[meth_idx], alpha = 0.6, color = col_set[meth_idx], linewidth = 2)
#     plt.scatter(ampLis, cnt_dct[meth[0]+'_'+meth[1]]-cen_power_list, alpha = 0.6, color = col_set[meth_idx])

# ax.set_xlabel('number of sites')
# ax.set_ylabel('power')
# plt.ylim(0.8, 1)
# ax.legend()
# plt.savefig('rewrite_power_10_amp.png')

# ampLis = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0]
# numSites = 8
# fnm = 'centralized_withNoise'
# cen_power_list = []
# for idx, amplification in enumerate(ampLis):
#     cen_power_list.append(get_power(os.path.join(dir_name, methLis[0][0], cnt_nm[0]+'_'+fnm+'_share'+str(numSites)+'_amp'+str(amplification)+'.csv'), 
#     os.path.join(dir_name, methLis[0][0], cnt_nm[1]+'_'+fnm+'_share'+str(numSites)+'_amp'+str(amplification)+'.csv'), 
#     os.path.join(dir_name, methLis[0][0], cnt_nm[2]+'_'+fnm+'_share'+str(numSites)+'_amp'+str(amplification)+'.csv'), 0.1))
# cen_power_list = np.array(cen_power_list)
# fig, ax = plt.subplots(figsize = (10, 6))
# plt.plot(ampLis, np.repeat(np.mean(cen_power_list), len(ampLis)), label = 'centralized', alpha = 0.3, color = 'k', linewidth = 2)
# cen_power_list -= np.mean(cen_power_list)
# fnm = 'decentralized_withNoise'
# meth_power_list = []
# for meth_idx, meth in enumerate(methLis):
#     share_power_list = []
#     for idx, amplification in enumerate(ampLis):
#         share_power_list.append(get_power(os.path.join(dir_name, meth[0], cnt_nm[0]+'_'+fnm+'_share'+str(numSites)+'_amp'+str(amplification)+'.csv'), 
#         os.path.join(dir_name, meth[0], cnt_nm[1]+'_'+fnm+'_share'+str(numSites)+'_amp'+str(amplification)+'.csv'), 
#         os.path.join(dir_name, meth[0], cnt_nm[2]+'_'+fnm+'_share'+str(numSites)+'_amp'+str(amplification)+'.csv'), 0.1))
#     meth_power_list.append(share_power_list)

# cnt_dct = {methLis[i][0]+'_'+methLis[i][1]: meth_power_list[i] for i in range(len(methLis))}

# for meth_idx, meth in enumerate(methLis):
#     plt.plot(ampLis, cnt_dct[meth[0]+'_'+meth[1]]-cen_power_list, label = lbLis[meth_idx], alpha = 0.6, color = col_set[meth_idx], linewidth = 2)
#     plt.scatter(ampLis, cnt_dct[meth[0]+'_'+meth[1]]-cen_power_list, alpha = 0.6, color = col_set[meth_idx])

# ax.set_xlabel('multiplier')
# ax.set_ylabel('power')
# plt.ylim(0.8, 1)
# ax.legend()
# plt.savefig('rewrite_power_8_amp.png')


dir_name = '~/amp_con/'
ampLis = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05]
numSites = 8
fnm = 'centralized_withNoise'
cen_power_list = []
for idx, amplification in enumerate(ampLis):
    cen_power_list.append(get_power(os.path.join(dir_name, methLis[0][0], cnt_nm[0]+'_'+fnm+'_share'+str(numSites)+'_amp'+str(amplification)+'.csv'), 
    os.path.join(dir_name, methLis[0][0], cnt_nm[1]+'_'+fnm+'_share'+str(numSites)+'_amp'+str(amplification)+'.csv'), 
    os.path.join(dir_name, methLis[0][0], cnt_nm[2]+'_'+fnm+'_share'+str(numSites)+'_amp'+str(amplification)+'.csv'), 0.1))
cen_power_list = np.array(cen_power_list)
fig, ax = plt.subplots(figsize = (20, 12))
plt.plot(ampLis, np.repeat(np.mean(cen_power_list), len(ampLis)), label = 'Centralized', alpha = 0.3, color = 'k', linewidth = 6)
cen_power_list -= np.mean(cen_power_list)
fnm = 'decentralized_withNoise'
meth_power_list = []
for meth_idx, meth in enumerate(methLis):
    share_power_list = []
    for idx, amplification in enumerate(ampLis):
        share_power_list.append(get_power(os.path.join(dir_name, meth[0], cnt_nm[0]+'_'+fnm+'_share'+str(numSites)+'_amp'+str(amplification)+'.csv'), 
        os.path.join(dir_name, meth[0], cnt_nm[1]+'_'+fnm+'_share'+str(numSites)+'_amp'+str(amplification)+'.csv'), 
        os.path.join(dir_name, meth[0], cnt_nm[2]+'_'+fnm+'_share'+str(numSites)+'_amp'+str(amplification)+'.csv'), 0.1))
    meth_power_list.append(share_power_list)

cnt_dct = {methLis[i][0]+'_'+methLis[i][1]: meth_power_list[i] for i in range(len(methLis))}

for meth_idx, meth in enumerate(methLis):
    plt.plot(ampLis, cnt_dct[meth[0]+'_'+meth[1]]-cen_power_list, label = lbLis[meth_idx], alpha = 0.6, color = col_set[meth_idx], linewidth = 6)
    plt.scatter(ampLis, cnt_dct[meth[0]+'_'+meth[1]]-cen_power_list, alpha = 0.6, color = col_set[meth_idx], s = 80)
desired_order = ['Centralized', 'Stouffer', 'Fisher', 'Pearson', 'Tippett', 'Largest Site']
handles, labels = plt.gca().get_legend_handles_labels()
ordered_handles = [h for l, h in sorted(zip(labels, handles), key=lambda lh: desired_order.index(lh[0]))]

ax.set_xlabel('Multiplier', fontsize=40)
ax.set_ylabel('Power', fontsize=40)
plt.ylim(0.834, 1)
plt.legend(ordered_handles, desired_order, fontsize=40, loc = 'lower center')
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.tick_params(direction='out', length=10, width=4)
plt.savefig('rewrite_power_8_amp.png')
