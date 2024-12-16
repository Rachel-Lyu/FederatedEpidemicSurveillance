import pandas as pd
import numpy as np
from scipy.special import ndtri
import scipy.stats as stats
import matplotlib.pyplot as plt
import scipy.stats as stats
import argparse
import os

def Stouffer(pvals, weights):
    Sstat = np.dot(weights, ndtri(pvals))
    return stats.norm.cdf(Sstat, scale = 1)

def Fisher(pvals, weights):
    Sstat = np.dot(weights, np.log(pvals))*len(pvals)
    return stats.chi2.sf(-2*Sstat, 2*len(pvals))

def wFisher(pvals, weights):
    Xstat = np.sum([stats.gamma.isf(pvals[i], weights[i]*len(pvals), loc=0, scale=2) for i in range(len(pvals))])
    return stats.gamma.sf(Xstat, len(pvals), loc=0, scale=2)

def Pearson(pvals, weights):
    Sstat = np.dot(weights, np.log(1-pvals))*len(pvals)
    return stats.chi2.cdf(-2*Sstat, 2*len(pvals))

def Tippett(pvals):
    Sstat = np.nanmin(pvals)
    return stats.beta.cdf(Sstat, 1, len(pvals))

def LargestSite(pvals, weights):
    return pvals[weights==weights.max()][0]

def CorrectedStouffer(pvals, weights, totalCounts, pi):
    Sstat = np.dot(weights, ndtri(pvals))
    if totalCounts > 0:
        Sstat += (1-len(pvals))/(2*np.sqrt(totalCounts*pi*(1-pi)))
    return stats.norm.cdf(Sstat, scale = 1)

def combinePval(pvals, combMethod = 'Stouffer', weightMethod = 'unweighted', shares = None, totalCounts = None, pi = None):
    pvals = np.array(pvals)
    pvals[pvals==0] = 1e-16
    pvals[pvals==1] = 1-1e-16
    if weightMethod == 'unweighted': 
        N = len(pvals)
        weights = np.repeat(1/N, N)
    elif weightMethod == 'default': 
        shares = np.array(shares)
        pvals = pvals[shares!=0]
        shares = shares[shares!=0]
        shares /= shares.sum()
        if combMethod in ['Stouffer', 'CorrectedStouffer']:
            weights = np.sqrt(shares)
        else:
            weights = shares
    else:
        weightMethod = np.array(weightMethod)
        if combMethod in ['Stouffer', 'CorrectedStouffer']:
            weights = weightMethod/np.sqrt(np.dot(weightMethod, weightMethod))
        else:
            weights = weightMethod/weightMethod.sum()

    if combMethod == 'Stouffer':
        return(Stouffer(pvals, weights))
    elif combMethod == 'Fisher':
        return(Fisher(pvals, weights))
    elif combMethod == 'wFisher':
        return(wFisher(pvals, weights))
    elif combMethod == 'Pearson':
        return(Pearson(pvals, weights))
    elif combMethod == 'LargestSite':
        return(LargestSite(pvals, weights))
    elif combMethod == 'Tippett':
        return(Tippett(pvals))
    elif combMethod == 'CorrectedStouffer':
        return(CorrectedStouffer(pvals, weights, totalCounts, pi))
    else:
        raise NotImplementedError('Undefined method')

# k: total counts
# l: relative length
def power(k, l, null_theta, real_theta, alpha_):
    kT_critical_value = stats.binom.ppf(1-alpha_, k, (1+null_theta)/(1+null_theta+l))
    return stats.binom.sf(kT_critical_value, k, (1+real_theta)/(1+real_theta+l))

def power_combineP_multinomial(k, l, null_theta, real_theta, alpha_, shares, combMethod, weightMethod, trial = 10000):
    success_cnt = 0
    for t_idx in range(trial):
        k_T = stats.binom.rvs(k, (1+real_theta)/(1+real_theta+l))
        k_i = stats.multinomial.rvs(k, shares)
        k_Ti = stats.multinomial.rvs(k_T, shares)
        pvals = np.array([stats.binom.sf(k_Ti[i]-1, k_i[i], (1+null_theta)/(1+null_theta+l)) for i in range(len(shares))])
        pval = combinePval(pvals, combMethod, weightMethod, shares, totalCounts = k_T, pi = (1+null_theta)/(1+null_theta+l))
        if pval < alpha_:
            success_cnt += 1
    return success_cnt/trial


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulate')
    parser.add_argument('--repeats',type=int,default=10,help='repeats = 10')
    args = parser.parse_args()
    alpha_ = 0.05
    null_theta = 0.3
    l = 2
    itv = 20
    kLis = np.arange(100, 1000+itv, itv)
    dir_name = 'test_bs_pthr'

    repeats = args.repeats
    k = 200
    idx_k = np.where(kLis == k)
    alpha_ = 0.05
    new_dir_name = 'power_curve'

    real_theta = np.linspace(0, 1.3, 50)
    BinomialPower = power(k = k, l = l, null_theta = null_theta, real_theta = real_theta, alpha_ = alpha_)
    with open(os.path.join(new_dir_name, 'BinomialPower.csv'), 'w') as f_cp:
        np.savetxt(f_cp, BinomialPower.reshape(1, -1), fmt='%.8f', delimiter=',')
        
    for r_idx in range(repeats):
        s_arr = [0.6, 0.8, 0.9, 2.0, 3.0, 5.0, 8.0, 10.0]
        np.random.shuffle(s_arr)
        for share in s_arr:
            if share < 1:
                methLis = [['Stouffer', 'unweighted'], ['Fisher', 'unweighted'], ['Stouffer', 'default'], ['Fisher', 'default'], ['wFisher', 'default'], ['Pearson', 'unweighted'], ['Tippett', 'unweighted'], ['LargestSite', 'default'], ['CorrectedStouffer', 'default']]
                lblis = ['Unweighted Stouffer', 'Unweighted Fisher', 'Weighted Stouffer', 'Weighted Fisher', 'wFisher', 'Pearson', 'Tippett', 'largest site', 'corrected Stouffer']
                shares = np.array([share, 1-share])
            elif share > 1:
                methLis = [['Stouffer', 'unweighted'], ['Fisher', 'unweighted'], ['Pearson', 'unweighted'], ['Tippett', 'unweighted'], ['LargestSite', 'unweighted'], ['CorrectedStouffer', 'unweighted']]
                lblis = ['Stouffer', 'Fisher', 'Pearson', 'Tippett', 'largest site', 'corrected Stouffer']
                shares = np.repeat(1/share, share)
            
            for m_idx, meth in enumerate(methLis):
                recArr = np.array(pd.read_csv(os.path.join(dir_name, 's'+str(share)+meth[0]+'_'+meth[1]+'.csv'), index_col=None, header=None)).mean(axis = 0)
                adj_alpha = recArr[idx_k]
                combineP = np.array([power_combineP_multinomial(k = k, l = l, null_theta = null_theta, real_theta = r, alpha_ = adj_alpha, shares = shares, combMethod = meth[0], weightMethod = meth[1]) for r in real_theta])
                with open(os.path.join(new_dir_name, 's'+str(share)+meth[0]+'_'+meth[1]+'_k'+str(k)+'.csv'), 'a+') as f_cp:
                    np.savetxt(f_cp, combineP.reshape(1, -1), fmt='%.4f', delimiter=',')
