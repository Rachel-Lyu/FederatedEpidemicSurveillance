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
col_set = [Category10[10][0], Category10[10][1], Category10[10][2], Category10[10][3], Category10[10][4]]
cnt_nm = ['TP', 'FN', 'FP']
methLis = [['Stouffer', 'unweighted'], ['Fisher', 'unweighted'], ['LargestSite', 'default'], ['Pearson', 'unweighted'], ['Tippett', 'unweighted']]
# lbLis = ['Stouffer\'s Z', 'Fisher\'s log(p)', 'Largest Site', 'Pearson\'s log(1-p)', 'Tippett\'s min(p)']
lbLis = ['Stouffer', 'Fisher', 'Largest Site', 'Pearson', 'Tippett']
desired_order = ['Centralized', 'Stouffer', 'Fisher', 'Pearson', 'Tippett', 'Largest Site']
shareList = ['[0.05 0.8 ]','[0.1  0.75]', '[0.1 0.7]','[0.1  0.65]', '[0.1 0.6]', '[0.15 0.55]',
             '[0.2 0.5]','[0.3 0.4]', '[0.2 0.4]', '[0.3 0.3]','[0.2 0.3]', '[0.2 0.2]']
entropyLis = np.array([0.483188, 0.556331, 0.627401, 0.69625 , 0.762707, 0.810316,
                       0.844541, 0.881353, 0.913865, 0.934978, 0.967489, 1.      ])
fnm = 'centralized_withNoise'
cen_power_list = []
for idx, share in enumerate(shareList):
    cen_power_list.append(get_power(os.path.join(dir_name, methLis[0][0]+'_'+methLis[0][1], cnt_nm[0]+'_'+fnm+'_share'+share+'.csv'), 
    os.path.join(dir_name, methLis[0][0]+'_'+methLis[0][1], cnt_nm[1]+'_'+fnm+'_share'+share+'.csv'), 
    os.path.join(dir_name, methLis[0][0]+'_'+methLis[0][1], cnt_nm[2]+'_'+fnm+'_share'+share+'.csv'), 0.1))
cen_power_list = np.array(cen_power_list)
fig, ax = plt.subplots(figsize = (20, 12))
plt.plot(entropyLis, np.repeat(np.mean(cen_power_list), len(entropyLis)), label = 'Centralized', alpha = 0.3, color = 'k', linewidth = 6)
cen_power_list -= np.mean(cen_power_list)
fnm = 'decentralized_withNoise'
meth_power_list = []
for meth_idx, meth in enumerate(methLis):
    share_power_list = []
    for idx, share in enumerate(shareList):
        share_power_list.append(get_power(os.path.join(dir_name, meth[0]+'_'+meth[1], cnt_nm[0]+'_'+fnm+'_share'+share+'.csv'), 
        os.path.join(dir_name, meth[0]+'_'+meth[1], cnt_nm[1]+'_'+fnm+'_share'+share+'.csv'), 
        os.path.join(dir_name, meth[0]+'_'+meth[1], cnt_nm[2]+'_'+fnm+'_share'+share+'.csv'), 0.1))
    meth_power_list.append(share_power_list)

cnt_dct = {methLis[i][0]+'_'+methLis[i][1]: meth_power_list[i] for i in range(len(methLis))}

for meth_idx, meth in enumerate(methLis):
    plt.plot(entropyLis, cnt_dct[meth[0]+'_'+meth[1]]-cen_power_list, label = lbLis[meth_idx], alpha = 0.6, color = col_set[meth_idx], linewidth = 6)
    plt.scatter(entropyLis, cnt_dct[meth[0]+'_'+meth[1]]-cen_power_list, alpha = 0.6, color = col_set[meth_idx], s = 80)
handles, labels = plt.gca().get_legend_handles_labels()
ordered_handles = [h for l, h in sorted(zip(labels, handles), key=lambda lh: desired_order.index(lh[0]))]
ax.set_xlabel('Normalized Entropy', fontsize=40)
ax.set_ylabel('Power', fontsize=40)
plt.ylim(0.9, 1)
plt.legend(ordered_handles, desired_order, fontsize=40)
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.tick_params(direction='out', length=10, width=4)
plt.savefig('rewrite_power_5S.png')

methLis = [['Stouffer', 'unweighted'], ['Fisher', 'unweighted'], ['LargestSite', 'default'], ['Stouffer', 'default'], ['wFisher', 'default'], ['CorrectedStouffer', 'default']]
lbLis = ['Unweighted Stouffer', 'Unweighted Fisher', 'Largest Site', 'Weighted Stouffer', 'wFisher', 'Corrected Stouffer']
desired_order = ['Centralized', 'Unweighted Stouffer', 'Unweighted Fisher', 'Weighted Stouffer', 'wFisher', 'Corrected Stouffer', 'Largest Site']
lstLis = ['-', '-', '-', '--', '--', ':']
lwdLis = [5.5, 5.5, 5.5, 6, 6, 7]
col_set = [Category10[10][0], Category10[10][1], Category10[10][2], Category10[10][0], Category10[10][1], Category10[10][0]]
fnm = 'centralized_withNoise'
cen_power_list = []
for idx, share in enumerate(shareList):
    cen_power_list.append(get_power(os.path.join(dir_name, meth[0]+'_'+meth[1], cnt_nm[0]+'_'+fnm+'_share'+share+'.csv'), 
    os.path.join(dir_name, meth[0]+'_'+meth[1], cnt_nm[1]+'_'+fnm+'_share'+share+'.csv'), 
    os.path.join(dir_name, meth[0]+'_'+meth[1], cnt_nm[2]+'_'+fnm+'_share'+share+'.csv'), 0.1))
cen_power_list = np.array(cen_power_list)
fig, ax = plt.subplots(figsize = (20, 12))
plt.plot(entropyLis, np.repeat(np.mean(cen_power_list), len(entropyLis)), label = 'Centralized', alpha = 0.3, color = 'k', linewidth = 6)
cen_power_list -= np.mean(cen_power_list)
fnm = 'decentralized_withNoise'
meth_power_list = []
for meth_idx, meth in enumerate(methLis):
    share_power_list = []
    for idx, share in enumerate(shareList):
        share_power_list.append(get_power(os.path.join(dir_name, meth[0]+'_'+meth[1], cnt_nm[0]+'_'+fnm+'_share'+share+'.csv'), 
        os.path.join(dir_name, meth[0]+'_'+meth[1], cnt_nm[1]+'_'+fnm+'_share'+share+'.csv'), 
        os.path.join(dir_name, meth[0]+'_'+meth[1], cnt_nm[2]+'_'+fnm+'_share'+share+'.csv'), 0.1))
    meth_power_list.append(share_power_list)

cnt_dct = {methLis[i][0]+'_'+methLis[i][1]: meth_power_list[i] for i in range(len(methLis))}

for meth_idx, meth in enumerate(methLis):
    plt.plot(entropyLis, cnt_dct[meth[0]+'_'+meth[1]]-cen_power_list, label = lbLis[meth_idx], alpha = 0.6, color = col_set[meth_idx], linestyle = lstLis[meth_idx], linewidth = lwdLis[meth_idx])
    plt.scatter(entropyLis, cnt_dct[meth[0]+'_'+meth[1]]-cen_power_list, alpha = 0.6, color = col_set[meth_idx], s = 80)
handles, labels = plt.gca().get_legend_handles_labels()
ordered_handles = [h for l, h in sorted(zip(labels, handles), key=lambda lh: desired_order.index(lh[0]))]
ax.set_xlabel('Normalized Entropy', fontsize=40)
ax.set_ylabel('Power', fontsize=40)
plt.ylim(0.9, 1)
plt.legend(ordered_handles, desired_order, fontsize=40)
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.tick_params(direction='out', length=10, width=4)
plt.savefig('rewrite_power_5S_wt.png')


dir_name = '~/numSites/'
col_set = Category10[10]
cnt_nm = ['TP', 'FN', 'FP']
methLis = [['Stouffer', 'unweighted'], ['Fisher', 'unweighted'], ['LargestSite', 'unweighted'], ['Pearson', 'unweighted'], ['Tippett', 'unweighted']]
# lbLis = ['Stouffer\'s Z', 'Fisher\'s log(p)', 'Largest Site', 'Pearson\'s log(1-p)', 'Tippett\'s min(p)']
lbLis = ['Stouffer', 'Fisher', 'Largest Site', 'Pearson', 'Tippett']
desired_order = ['Centralized', 'Stouffer', 'Fisher', 'Pearson', 'Tippett', 'Largest Site']
numSitesLis = [2, 3, 5, 8, 10, 15, 20]
fnm = 'centralized_withNoise'
cen_power_list = []
for idx, numSites in enumerate(numSitesLis):
    cen_power_list.append(get_power(os.path.join(dir_name, methLis[0][0], cnt_nm[0]+'_'+fnm+'_share'+str(numSites)+'.csv'), 
    os.path.join(dir_name, methLis[0][0], cnt_nm[1]+'_'+fnm+'_share'+str(numSites)+'.csv'), 
    os.path.join(dir_name, methLis[0][0], cnt_nm[2]+'_'+fnm+'_share'+str(numSites)+'.csv'), 0.1))
cen_power_list = np.array(cen_power_list)
fig, ax = plt.subplots(figsize = (20, 12))
plt.plot(numSitesLis, np.repeat(np.mean(cen_power_list), len(numSitesLis)), label = 'Centralized', alpha = 0.3, color = 'k', linewidth = 6)
cen_power_list -= np.mean(cen_power_list)
fnm = 'decentralized_withNoise'
meth_power_list = []
for meth_idx, meth in enumerate(methLis):
    share_power_list = []
    for idx, numSites in enumerate(numSitesLis):
        share_power_list.append(get_power(os.path.join(dir_name, meth[0], cnt_nm[0]+'_'+fnm+'_share'+str(numSites)+'.csv'), 
        os.path.join(dir_name, meth[0], cnt_nm[1]+'_'+fnm+'_share'+str(numSites)+'.csv'), 
        os.path.join(dir_name, meth[0], cnt_nm[2]+'_'+fnm+'_share'+str(numSites)+'.csv'), 0.1))
    meth_power_list.append(share_power_list)

cnt_dct = {methLis[i][0]+'_'+methLis[i][1]: meth_power_list[i] for i in range(len(methLis))}

for meth_idx, meth in enumerate(methLis):
    plt.plot(numSitesLis, cnt_dct[meth[0]+'_'+meth[1]]-cen_power_list, label = lbLis[meth_idx], alpha = 0.6, color = col_set[meth_idx], linewidth = 6)
    plt.scatter(numSitesLis, cnt_dct[meth[0]+'_'+meth[1]]-cen_power_list, alpha = 0.6, color = col_set[meth_idx], s = 80)
handles, labels = plt.gca().get_legend_handles_labels()
ordered_handles = [h for l, h in sorted(zip(labels, handles), key=lambda lh: desired_order.index(lh[0]))]
ax.set_xlabel('Number of Sites', fontsize=40)
ax.set_ylabel('Power', fontsize=40)
plt.ylim(0.5, 1)
plt.xticks([2, 3, 5, 8, 10, 15, 20], [2, 3, 5, 8, 10, 15, 20])
plt.legend(ordered_handles, desired_order, fontsize=40)
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.tick_params(direction='out', length=10, width=4)
plt.savefig('rewrite_power_nS.png')