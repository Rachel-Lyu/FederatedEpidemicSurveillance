import pandas as pd
import numpy as np
import datetime
import os 
from scipy.special import ndtri
import scipy.stats as stats
from collections import OrderedDict
import matplotlib.pyplot as plt

# Unweighted
# def SF_df2n(pvals):
def SF(pvals):
    pvals[pvals==0] = 1e-10
    pvals[pvals==1] = 1-1e-10
    Sstat = np.sum(np.log(pvals))
    return stats.chi2.sf(-2*Sstat, 2*len(pvals))

def SP(pvals):
    pvals[pvals==0] = 1e-10
    pvals[pvals==1] = 1-1e-10
    Sstat = np.sum(np.log(1-pvals))
    return stats.chi2.cdf(-2*Sstat, 2*len(pvals))

def SS(pvals):
    pvals[pvals==0] = 1e-10
    pvals[pvals==1] = 1-1e-10
    Sstat = np.sum(ndtri(pvals))
    return stats.norm.cdf(Sstat, scale = np.sqrt(len(pvals)))

def ST(pvals):
    # pvals[pvals==0] = 1e-10
    Sstat = np.nanmin(pvals)
    return stats.beta.cdf(Sstat, 1, len(pvals))

def largestSite(pvals, shares):
    return np.nanmean(pvals[shares==shares.max()])

# Weighted
# wt = share
def SF_wt(pvals, shares):
    wt = np.array(shares)
    pvals[pvals==0] = 1e-10
    pvals[pvals==1] = 1-1e-10
    Sstat = np.dot(wt, np.log(pvals))*len(pvals)
    return stats.chi2.sf(-2*Sstat, 2*len(pvals))

def SP_wt(pvals, shares):
    wt = np.array(shares)
    pvals[pvals==0] = 1e-10
    pvals[pvals==1] = 1-1e-10
    Sstat = np.dot(wt, np.log(1-pvals))*len(pvals)
    return stats.chi2.cdf(-2*Sstat, 2*len(pvals))

# wt = np.sqrt(shares)
def SS_wt(pvals, shares):
    wt = np.array(shares)
    pvals[pvals==0] = 1e-10
    pvals[pvals==1] = 1-1e-10
    wt = np.sqrt(wt)
    Sstat = np.dot(wt, ndtri(pvals))
    return stats.norm.cdf(Sstat, scale = 1)

# With estimated sum
def SS_correction(pvals, shares, sum_, pi):
    wt = np.array(shares)
    pvals[pvals==0] = 1e-10
    pvals[pvals==1] = 1-1e-10
    wt = np.sqrt(wt)
    Sstat = np.dot(wt, ndtri(pvals))
    Sstat += (1-len(pvals))/(2*np.sqrt(sum_*pi*(1-pi)))
    return stats.norm.cdf(Sstat, scale = 1)

def combinePval(combinedPmethod, pvals, shares = None, sum_ = None, pi = None):
    if shares is not None:
        pvals = pvals[shares!=0]
        shares = shares[shares!=0]
    if combinedPmethod in ['SF', 'SS', 'SP', 'ST', 'largestSite', 'SF_wt', 'SP_wt', 'SS_wt', 'SS_correction']:
        # Unweighted
        if combinedPmethod == 'SF':
            return(SF(pvals))
        if combinedPmethod == 'SS':
            return(SS(pvals))
        if combinedPmethod == 'SP':
            return(SP(pvals))
        # Select one
        if combinedPmethod == 'ST':
            return(ST(pvals))
        if combinedPmethod == 'largestSite':
            return(largestSite(pvals, shares))
        # Weighted
        if combinedPmethod == 'SF_wt':
            return(SF_wt(pvals, shares))
        if combinedPmethod == 'SP_wt':
            return(SP_wt(pvals, shares))
        if combinedPmethod == 'SS_wt':
            return(SS_wt(pvals, shares))
        # With estimated sum(n)
        if combinedPmethod == 'SS_correction':
            return(SS_correction(pvals, shares, sum_, pi))
        else:
            raise NotImplementedError('Undefined method ' + combinedPmethod + ' (my fault)')
    else:
        raise NotImplementedError('Undefined method ' + combinedPmethod + ' (your fault)')

class cenDecen:
    def __init__(self, date_value, subTS, allocProp = None, growthRateThr = 20, pval_thr = 0.05, case_thr = 20, combinedPmethod = 'SS', window4Pval = 2, updating_period = 2, training_period = 4, share_lag = 1, rep_per_wk = 1):
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
        self.combinedPmethod = combinedPmethod
        self.window4Pval = window4Pval

        self.computeGrowthRate()
        self.growthAlarm = self.alarmMask_once_d_with_thr(self.window4Pval*self.rep_per_wk, self.growthRate, self.growthRateThr)
        self.pval = self.computePval(self.TimeSeries)
        self.pvalAlarm = self.alarmMask_once_d_with_thr(self.window4Pval*self.rep_per_wk, self.pval, self.pval_thr, ge_thr = False)
        
        self.pval_subTS = [self.computePval(self.subTS[i]) for i in range(self.numFclt)]
        self.get_combinePval(allocProp is None)
        self.combinedPAlarm = self.alarmMask_once_d_with_thr(self.window4Pval*self.rep_per_wk, self.combinedP, self.pval_thr, ge_thr = False)

    def update_share(self, date_idx):
        self.allocProp = np.nanmean(self.subTS[:, np.max([0, date_idx - self.training_period*self.rep_per_wk]):date_idx], axis=1)
        self.allocProp /= np.nansum(self.allocProp)
        return

    def update_sum(self, date_idx):
        sum_ = np.nansum(self.TimeSeries[(date_idx - self.window4Pval*self.rep_per_wk):(date_idx+1)])
        return sum_

    def get_interval_index(self, end_index):
        arr = np.array(self.date_value)
        return np.where(np.logical_and(arr < arr[end_index], arr >= arr[end_index] - datetime.timedelta(weeks=self.window4Pval)))[0]

    def computePval(self, signal):
        pvalPoi = []
        for i in range(self.window4Pval*self.rep_per_wk, self.TSlen):
            idx_sel = self.get_interval_index(i)
            k = np.nansum(signal[idx_sel])
            n = k + signal[i]
            p = len(idx_sel)/(1+0.01*self.growthRateThr+len(idx_sel))
            pvalPoi.append(stats.binom.cdf(k, n, p))
        return pvalPoi

    def alarmMask_once_d_with_thr(self, TSshift, seq, thr, ge_thr = True):
        assert len(self.TimeSeries) - TSshift == len(seq), str(len(self.TimeSeries) - TSshift)+' != '+str(len(seq))
        if ge_thr:
            allSet = set(np.where(np.array(seq)>thr)[0])
        else:
            allSet = set(np.where(np.array(seq)<thr)[0])
        valid_case = set(np.where(np.array(self.TimeSeries[TSshift:])>self.case_thr)[0])
        allSet = allSet.intersection(valid_case)
        allSet = [(i + TSshift) for i in allSet]
        allSet.sort()
        return allSet

    def computeGrowthRate(self):
        self.growthRate = [(self.TimeSeries[i]/np.mean(self.TimeSeries[self.get_interval_index(i)])-1)*100 for i in range(self.window4Pval*self.rep_per_wk, self.TSlen)]
        return

    def get_combinePval(self, flag):
        upd_cnt = -self.share_lag
        combinedP = []
        sum_ = np.nansum(self.TimeSeries[((self.updating_period-self.window4Pval)*self.rep_per_wk):(self.updating_period*self.rep_per_wk+1)])
        for i in range(self.TSlen-self.window4Pval*self.rep_per_wk):
            testpval = np.array([self.pval_subTS[j][i] for j in range(self.numFclt)])
            combinedP.append(combinePval(self.combinedPmethod, testpval, self.allocProp, sum_, (self.window4Pval*self.rep_per_wk)/(1+0.01*self.growthRateThr+(self.window4Pval*self.rep_per_wk))))
            upd_cnt += 1
            if upd_cnt >= self.updating_period*self.rep_per_wk:
                if flag: 
                    self.update_share(i+1-self.share_lag)
                sum_ = self.update_sum(i+1-self.share_lag)
                upd_cnt = 0
        self.combinedP = combinedP
        return self.combinedP

def mappingAlarms(tgt, sim, beforeR, afterR):
    TPset = set()
    FPset = set(sim)
    FNset = set(tgt)
    trueTP = set()
    for tgt_item in tgt:
        flag = False
        simArr = np.array(sim)
        dist = tgt_item - simArr
        for d in dist:
            if d < beforeR and d > -afterR:
                ele = simArr[np.where(dist == d)][0]
                TPset.add(ele)
                if ele in FPset:
                    FPset.remove(ele)
                flag = True
        if flag:
            trueTP.add(tgt_item)
            FNset.remove(tgt_item)
    TPseq = list(TPset)
    TPseq.sort()
    FPseq = list(FPset)
    FPseq.sort()
    FNseq = list(FNset)
    FNseq.sort()
    trueTP = list(trueTP)
    trueTP.sort()
    return TPseq, FPseq, FNseq, trueTP

def genAlarmDict(TruePred, mappedTime, beforeR, afterR):
    trueTPdict = {}
    for i in mappedTime:
        dist = i - np.array(TruePred)
        temp = np.array([], dtype=int)
        for d in dist:
            if d < beforeR and d > -afterR:
                idx = np.where(dist == d)[0]
                temp = np.append(temp, TruePred[idx[0]])
            if d == dist[-1]:
                trueTPdict[i] = temp.tolist()
    return trueTPdict

def compLagDays(almDict, firstAlarm = True):
    if not almDict:
        return 0
    else:
        lag = np.array([])
        for tgt, sim in almDict.items():
            if firstAlarm:
                t, s = tgt, sim[0]
            else:
                t, s = tgt, np.mean(sim)
            lag = np.append(lag, s-t)
        return np.mean(lag)

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    else:
        print('File exists!')

if __name__ == '__main__':
    df_out = pd.read_csv('hospitalization_summary.csv')
    locSelLis = list(df_out['loc'])

    rep_per_wk = 1
    growthRateThr = 20
    window4Pval = 2
    pval_thr_ref = 0.05
    updating_period = 12
    training_period = 12
    share_lag = 2
    evalstart = np.max([updating_period, training_period])

    dict_out = {k: None for k in ['loc', 'numFclt', 'mean_cnts']}
    df_out = pd.DataFrame(dict_out, index=[])
    for loc_idx in range(len(locSelLis)):
        loc = locSelLis[loc_idx]
        TSdf = pd.read_csv('hospitalization/'+loc+'.csv', index_col=0)
        date_value = [datetime.datetime.strptime(dt,'%Y-%m-%d').date() for dt in TSdf['datetime']]
        TS = list(TSdf['total_cnts'])
        subTS = np.array(TSdf.iloc[:, 2:-1]).transpose()
        originalData = cenDecen(date_value, subTS, allocProp = None, growthRateThr = growthRateThr, pval_thr = pval_thr_ref, window4Pval = window4Pval, updating_period = updating_period, training_period = training_period, share_lag = share_lag, rep_per_wk = rep_per_wk)
        ref_alarm = originalData.alarmMask_once_d_with_thr((originalData.window4Pval+evalstart)*originalData.rep_per_wk, originalData.pval[evalstart*originalData.rep_per_wk:], originalData.pval_thr, ge_thr = False)
        if len(ref_alarm) > 0:
            df_out.loc[len(df_out.index)] = [loc, originalData.numFclt, np.nanmean(TS)]
    df_out.to_csv('hospitalization_result_summary.csv')