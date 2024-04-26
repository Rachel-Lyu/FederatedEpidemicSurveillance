import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os 
from scipy.special import ndtri
from collections import OrderedDict

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

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    else:
        print('File exists!')

def mappingAlarms(tgt, sim, beforeR, afterR):
    tgtArr = np.array(tgt)
    simArr = np.array(sim)
    dist = np.subtract.outer(tgtArr, simArr)
    within_range = np.logical_and(dist < beforeR, dist > -afterR)
    mapped_sim = np.bool_(1 - (~within_range).prod(axis=0))
    mapped_tgt = np.bool_(1 - (~within_range).prod(axis=1))
    TPseq = simArr[mapped_sim].tolist()
    trueTP = tgtArr[mapped_tgt].tolist()
    FPseq = np.setdiff1d(sim, TPseq).tolist()
    FNseq = np.setdiff1d(tgt, trueTP).tolist()
    return TPseq, FPseq, FNseq, trueTP

class cenDecen:
    def __init__(self, date_value, subTS, allocProp = None, growthRateThr = 20, pval_thr = 0.05, case_thr = 20, combMethod = 'Stouffer', weightMethod = 'default', window4Pval = 2, updating_period = 2, training_period = 4, share_lag = 1, rep_per_wk = 1):
        self.rep_per_wk = rep_per_wk
        self.updating_period = updating_period
        self.training_period = training_period
        self.share_lag = share_lag
        if allocProp is None:
            self.date_value = date_value[(self.updating_period-window4Pval)*self.rep_per_wk:]
            self.subTS = subTS[:, (self.updating_period-window4Pval)*self.rep_per_wk:]
            self.allocProp = subTS[:, :self.updating_period*self.rep_per_wk].mean(axis=1)
        else:
            self.date_value = date_value
            self.subTS = subTS
            self.allocProp = allocProp
        
        self.allocProp /= np.sum(self.allocProp)
        self.numFclt = len(self.subTS)
        self.TimeSeries = np.rint(self.subTS.sum(axis=0))
        self.TSlen = len(self.TimeSeries)

        self.growthRateThr = growthRateThr
        self.pval_thr = pval_thr
        self.case_thr = case_thr
        self.window4Pval = window4Pval
        self.combMethod = combMethod
        self.weightMethod = weightMethod
        self.l_shift = self.window4Pval*self.rep_per_wk

        self.computeGrowthRate()
        self.growthAlarm = self.alarmMask_once_d_with_thr(self.l_shift, self.growthRate, self.growthRateThr)
        self.pval = self.computePval(self.TimeSeries)
        self.pvalAlarm = self.alarmMask_once_d_with_thr(self.l_shift, self.pval, self.pval_thr, ge_thr = False)
        
        self.pval_subTS = [self.computePval(self.subTS[i]) for i in range(self.numFclt)]
        self.get_combinePval(change_share = allocProp is None)
        self.combinedPAlarm = self.alarmMask_once_d_with_thr(self.l_shift, self.combinedP, self.pval_thr, ge_thr = False)

    def update_share(self, date_idx):
        tmp = np.nanmean(self.subTS[:, np.max([0, date_idx - self.training_period*self.rep_per_wk]):date_idx], axis=1)
        tmp /= np.nansum(tmp)
        if np.sum(np.isnan(tmp)) == 0:
            self.allocProp = tmp
        return

    def update_sum(self, date_idx):
        sum_ = np.nansum(self.TimeSeries[(date_idx - self.l_shift):(date_idx+1)])
        return sum_

    def get_interval_index(self, end_index):
        arr = np.array(self.date_value)
        return np.where(np.logical_and(arr < arr[end_index], arr >= arr[end_index] - datetime.timedelta(weeks=self.window4Pval)))[0]

    def computePval(self, signal):
        pvalPoi = np.empty(self.TSlen - self.l_shift, dtype=float)
        for i in range(self.l_shift, self.TSlen):
            idx_sel = self.get_interval_index(i)
            k = np.nansum(signal[idx_sel])
            n = k + signal[i]
            p = len(idx_sel)/(1+0.01*self.growthRateThr+len(idx_sel))
            pvalPoi[i - self.l_shift] = stats.binom.cdf(k, n, p)
        return pvalPoi

    def alarmMask_once_d_with_thr(self, TSshift, seq, thr, ge_thr=True):
        if len(self.TimeSeries) - TSshift != len(seq):
            raise ValueError("Length of the input sequence does not match the length of the TimeSeries.")
        if ge_thr:
            all_indices = np.where(np.array(seq) > thr)[0]
        else:
            all_indices = np.where(np.array(seq) < thr)[0]
        valid_indices = np.where(np.array(self.TimeSeries[TSshift:]) > self.case_thr)[0]
        all_indices = np.intersect1d(all_indices, valid_indices)
        result_indices = all_indices + TSshift
        sorted_indices = np.argsort(result_indices)
        return result_indices[sorted_indices].tolist()

    def computeGrowthRate(self):
        self.growthRate = [(self.TimeSeries[i]/np.mean(self.TimeSeries[self.get_interval_index(i)])-1)*100 for i in range(self.l_shift, self.TSlen)]
        return

    def get_combinePval(self, change_share):
        upd_cnt = -self.share_lag
        combinedP = np.empty(self.TSlen - self.l_shift, dtype=float)
        sum_ = np.nansum(self.TimeSeries[((self.updating_period - self.window4Pval) * self.rep_per_wk):(self.updating_period * self.rep_per_wk + 1)])
        pi_ = self.l_shift/(1+0.01*self.growthRateThr+self.l_shift)
        for i in range(self.TSlen - self.l_shift):
            testpval = np.array(self.pval_subTS)[:, i]
            combinedP[i] = combinePval(testpval, combMethod = self.combMethod, weightMethod = self.weightMethod, shares = self.allocProp, totalCounts = sum_, pi = pi_)
            upd_cnt += 1
            if upd_cnt >= self.updating_period*self.rep_per_wk:
                if change_share: 
                    self.update_share(i+1-self.share_lag)
                sum_ = self.update_sum(i+1-self.share_lag)
                upd_cnt = 0
        self.combinedP = combinedP
        return self.combinedP

if __name__ == '__main__':
    df_out = pd.read_csv('claim_summary.csv')
    locSelLis = list(df_out['state'])

    rep_per_wk = 7
    growthRateThr = 30
    window4Pval = 2
    pval_thr_ref = 0.05
    # updating_period = 12
    # training_period = 12
    # share_lag = 2
    # evalstart = np.max([updating_period, training_period])
    evalstart = 12
    pthr = np.linspace(0.001, 0.5, 500)
    pthr = np.append([1e-8, 1e-6, 1e-4], pthr)
    pthr = np.append(pthr, np.linspace(0.51, 1.0, 50))
    methLis = [['Stouffer', 'unweighted'], ['Stouffer', 'default'], ['Fisher', 'unweighted'], ['Fisher', 'default'], ['wFisher', 'default'], ['Pearson', 'unweighted'], ['Tippett', 'unweighted'], ['LargestSite', 'default'], ['CorrectedStouffer', 'default']]
    dir_name = 'rewrite_lag_claim'

    np.random.seed(0)
    mkdir(dir_name)
    for loc_idx in range(len(locSelLis)):
        loc = locSelLis[loc_idx]
        TSdf = pd.read_csv('claim/'+loc+'.csv', index_col=0)
        date_value = [datetime.datetime.strptime(dt,'%Y-%m-%d').date() for dt in TSdf['date']]
        TS = list(TSdf['total_cnts'])
        subTS = np.array(TSdf.iloc[:, 2:-1]).transpose()
        for [updating_period, training_period, share_lag] in [[12, 12, 4], [4, 12, 4]]:
            originalData = cenDecen(date_value, subTS, allocProp = None, growthRateThr = growthRateThr, weightMethod = 'unweighted', pval_thr = pval_thr_ref, window4Pval = window4Pval, updating_period = updating_period, training_period = training_period, share_lag = share_lag, rep_per_wk = rep_per_wk)
            ref_alarm = originalData.alarmMask_once_d_with_thr((originalData.window4Pval+evalstart)*originalData.rep_per_wk, originalData.pval[evalstart*originalData.rep_per_wk:], originalData.pval_thr, ge_thr = False)
            if len(ref_alarm) > 0:
                for m_idx, meth in enumerate(methLis):
                    TP_PvsP = np.zeros((len(pthr),))
                    FP_PvsP = np.zeros((len(pthr),))
                    FN_PvsP = np.zeros((len(pthr),))
                    originalData.combMethod = meth[0]
                    originalData.weightMethod = meth[1]
                    originalData.get_combinePval(change_share = True)
                    for pv_idx in range(len(pthr)):
                        originalData.pval_thr = pthr[pv_idx]
                        originalData.combinedPAlarm = originalData.alarmMask_once_d_with_thr((originalData.window4Pval+evalstart)*originalData.rep_per_wk, originalData.combinedP[evalstart*originalData.rep_per_wk:], pthr[pv_idx], ge_thr = False)
                        new_alarm = originalData.combinedPAlarm
                        TPsq_tmp, FPsq_tmp, FNsq_tmp, mappedTime_tmp = mappingAlarms(ref_alarm, new_alarm, 1*rep_per_wk, 2*rep_per_wk)
                        TP_PvsP[pv_idx] = len(mappedTime_tmp)
                        FP_PvsP[pv_idx] = len(FPsq_tmp)
                        FN_PvsP[pv_idx] = len(FNsq_tmp)
                        
                    recall = TP_PvsP/(FN_PvsP+TP_PvsP)
                    precision = TP_PvsP/(FP_PvsP+TP_PvsP)
                    precision[np.isnan(precision)] = 1
                    with open(os.path.join(dir_name, 'updating'+str(updating_period)+'training'+str(training_period)+'lag'+str(share_lag)+meth[0]+'_'+meth[1]+'_recall.csv'), 'a+') as f_TP:
                        np.savetxt(f_TP, recall.reshape(1, -1), fmt='%.5f', delimiter=',')
                    with open(os.path.join(dir_name, 'updating'+str(updating_period)+'training'+str(training_period)+'lag'+str(share_lag)+meth[0]+'_'+meth[1]+'_precision.csv'), 'a+') as f_FP:
                        np.savetxt(f_FP, precision.reshape(1, -1), fmt='%.5f', delimiter=',')
