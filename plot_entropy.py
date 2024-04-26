import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def normalized_entropy(prob_vec):
    return np.round(-np.dot(prob_vec, np.log(prob_vec))/np.log(len(prob_vec)), 6)

df_out = pd.read_csv('claim_summary.csv', header=0, index_col=0)
df_out_ = pd.read_csv('claim_result_summary.csv', header=0, index_col=0)
df_out = df_out.merge(df_out_, how='inner', left_on='state', right_on='loc')
fig = plt.figure(figsize = (20, 12))
sc = plt.scatter(df_out['valid_cts'], df_out['mean_cnts'], c = df_out['normalized_entropy'], s = 200, alpha = 0.5, cmap='hsv')
plt.xlabel('Number of counties for aggregation', fontsize=40)
plt.ylabel('Average counts', fontsize=40)
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.tick_params(direction='out', length=10, width=4)
cbar = fig.colorbar(sc)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(30)
plt.clim(0.4, 1)
cbar.ax.tick_params(direction='out', length=8, width=3)
cbar.set_label("Entropy", fontsize=40)
plt.savefig('claim_summary.png')

df_out = pd.read_csv('hospitalization_summary.csv', header=0, index_col=0)
df_out_ = pd.read_csv('hospitalization_result_summary.csv', header=0, index_col=0)
df_out = df_out.merge(df_out_, how='inner', on=['loc', 'numFclt'])
fig = plt.figure(figsize = (20, 12))
sc = plt.scatter(df_out['numFclt'], df_out['mean_cnts'], c = df_out['normalized_entropy'], s = 200, alpha = 0.5, cmap='hsv')
plt.xlabel('Number of facilities for aggregation', fontsize=40)
plt.ylabel('Average counts', fontsize=40)
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.tick_params(direction='out', length=10, width=4)
cbar = fig.colorbar(sc)
for t in cbar.ax.get_yticklabels():
    t.set_fontsize(30)
cbar.ax.tick_params(direction='out', length=8, width=3)
    
plt.clim(0.4, 1)
cbar.set_label("Entropy", fontsize=40)
plt.savefig('hospitalization_summary.png')