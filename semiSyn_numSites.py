import pandas as pd
import numpy as np
import datetime
import os 
from scipy.special import ndtri
import scipy.stats as stats
import matplotlib.pyplot as plt
import scipy.stats as stats
import argparse
import itertools

def genProp(numSites, itv, lenUnique):
    all_prob = np.array(list(itertools.combinations_with_replacement(np.arange(itv, 1 - (numSites - 2) * itv, itv), numSites)))
    all_prob = all_prob[all_prob.sum(axis=1) == 1]
    return all_prob[[len(np.unique(v, axis=0)) < lenUnique for v in all_prob]]

def normalized_entropy(prob_vec):
    return np.round(-np.dot(prob_vec, np.log(prob_vec))/np.log(len(prob_vec)), 6)

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

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    else:
        print('File exists!')

def moving_avg_smoother(signal, window_length, mode = 'before'):
    if mode == 'before':
        signal_padded = np.append(np.nan * np.ones(window_length - 1), signal)
    if mode == 'center':
        bfLen = int(np.ceil((window_length - 1)/2))
        signal_padded = np.append(np.nan * np.ones(bfLen), signal)
        signal_padded = np.append(signal_padded, np.nan * np.ones(window_length - 1 - bfLen))
    signal_smoothed = (
        np.convolve(
            signal_padded, np.ones(window_length, dtype=int), mode='valid'
        )
        / window_length
    )
    if mode == 'before':
        signal_smoothed[:(window_length-1)] = signal[:(window_length-1)]
    if mode == 'center':
        signal_smoothed[:bfLen] = signal[:bfLen]
        signal_smoothed[-((window_length - 1 - bfLen)):] = signal[-((window_length - 1 - bfLen)):]
    return signal_smoothed
 
class SimAllocComb:
    def __init__(self, date_value, TS, growthRateThr = 30, pval_thr = 0.05, case_thr = 20, window4Pval = 2, rep_per_wk = 7):
        self.rep_per_wk = rep_per_wk
        self.date_value = date_value
        self.TimeSeries = np.rint(TS)
        self.TSlen = len(self.TimeSeries)
        self.growthRateThr = growthRateThr
        self.pval_thr = pval_thr
        self.case_thr = case_thr
        self.window4Pval = window4Pval
        self.l_shift = self.window4Pval*self.rep_per_wk
        
        self.pval = self.computePval(self.TimeSeries)
        self.pvalAlarm = self.alarmMask_once_d_with_thr(self.l_shift, self.pval, self.pval_thr, self.TimeSeries, ge_thr = False)
        self.growthAlarm = self.alarmMask_once_d_with_thr(self.l_shift, self.computeGrowth(self.TimeSeries), self.growthRateThr, self.TimeSeries, ge_thr = True)
        
    def get_interval_index(self, end_index):
        arr = np.array(self.date_value)
        return np.where(np.logical_and(arr < arr[end_index], arr >= arr[end_index] - datetime.timedelta(days=self.l_shift)))[0]

    def computePval(self, signal):
        pvalPoi = np.empty(self.TSlen - self.l_shift, dtype=float)
        for i in range(self.l_shift, self.TSlen):
            idx_sel = self.get_interval_index(i)
            k = np.nansum(signal[idx_sel])
            n = k + signal[i]
            p = len(idx_sel)/(1+0.01*self.growthRateThr+len(idx_sel))
            pvalPoi[i - self.l_shift] = stats.binom.cdf(k, n, p)
        return pvalPoi
    
    def computeGrowth(self, signal):
        growthArr = np.empty(self.TSlen - self.l_shift, dtype=float)
        for i in range(self.l_shift, self.TSlen):
            lam_B = np.nanmean(signal[self.get_interval_index(i)])
            if lam_B == 0:
                lam_B = 0.5
            growthArr[i - self.l_shift] = (signal[i]/lam_B - 1)*100
        return growthArr
    
    def alarmMask_once_d_with_thr(self, TSshift, seq, thr, ref_TS, ge_thr=True):
        if len(ref_TS) - TSshift != len(seq):
            raise ValueError("Length of the input sequence does not match the length of the TimeSeries.")
        if ge_thr:
            all_indices = np.where(np.array(seq) > thr)[0]
        else:
            all_indices = np.where(np.array(seq) < thr)[0]
        valid_indices = np.where(np.array(ref_TS[TSshift:]) > self.case_thr)[0]
        all_indices = np.intersect1d(all_indices, valid_indices)
        result_indices = all_indices + TSshift
        sorted_indices = np.argsort(result_indices)
        return result_indices[sorted_indices].tolist()

class simDat(SimAllocComb):
    def __init__(self, date_value, TS, allocProp, combMethod = 'Stouffer', weightMethod = 'default', growthRateThr = 30, pval_thr = 0.05, case_thr = 20, window4Pval = 2, rep_per_wk = 7):
        super().__init__(date_value, TS, growthRateThr, pval_thr, case_thr, window4Pval, rep_per_wk)
        self.allocProp = allocProp
        self.combMethod = combMethod
        self.weightMethod = weightMethod
        self.alloc2facilities()
        self.get_combinePval()
        self.combinedPAlarm = self.alarmMask_once_d_with_thr(self.l_shift, self.combinedP, self.pval_thr, self.TimeSeries, ge_thr = False)

    def genSubSeq(self, replacement = False):
        if replacement:
            return np.random.binomial(np.array(self.TimeSeries)[:, None], np.array(self.allocProp)[None, :]).T
        else:
            return np.array([np.random.multinomial(int_, self.allocProp).T for int_ in self.TimeSeries]).T

    def alloc2facilities(self):
        self.subTS = self.genSubSeq()
        self.pval_subTS = [self.computePval(self.subTS[i]) for i in range(len(self.allocProp))]
        return

    def get_combinePval(self):
        combinedP = np.empty(self.TSlen - self.l_shift, dtype=float)
        pi_ = self.l_shift/(1+0.01*self.growthRateThr+self.l_shift)
        for i in range(self.TSlen-self.l_shift):
            testpval = np.array(self.pval_subTS)[:, i]
            sum_ = np.nansum(self.TimeSeries[i:i+self.l_shift+1])
            combinedP[i] = combinePval(testpval, combMethod = self.combMethod, weightMethod = self.weightMethod, shares = self.allocProp, totalCounts = sum_, pi = pi_)
        self.combinedP = combinedP
        return self.combinedP
    
# underlying pattern as ground truth
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulate')
    parser.add_argument('--growthRateThr',type=float,default='30',help='growthRateThr = 30')
    parser.add_argument('--pval_thr',type=float,default=0.05,help='pval_thr = 0.05')
    parser.add_argument('--nLoops',type=int,default=100,help='nLoops = 100')
    parser.add_argument('--maxPthr',type=float,default=0.5,help='maxPthr = 0.5')
    parser.add_argument('--dir_name',type=str,default='test',help='dir_name = \'test\'')
    args = parser.parse_args()

    df = pd.read_csv('all_states.csv', index_col=0)
    date_value = [datetime.datetime.strptime(dt,'%Y-%m-%d').date() for dt in df.index]
    growthRateThr = args.growthRateThr
    pval_thr = args.pval_thr
    nLoops = args.nLoops
    maxPthr = args.maxPthr

    methLis = [['Stouffer', 'unweighted'], ['Fisher', 'unweighted'], ['Pearson', 'unweighted'], ['Tippett', 'unweighted'], ['LargestSite', 'unweighted'], ['CorrectedStouffer', 'unweighted']]

    smoother_window_length = 7
    window4Pval = 2
    rep_per_wk = 7
    bfLen = int(np.ceil((smoother_window_length - 1)/2))
    date_value = date_value[bfLen:-(smoother_window_length - 1 - bfLen)]
    nPs = int(maxPthr/0.005)
    pthr = np.linspace(0.005, maxPthr, nPs)
    pthr = np.append([1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3], pthr)
    nPs = len(pthr)
    mkdir(args.dir_name)
    # np.random.seed(0)
    for meth in methLis:
        mkdir(os.path.join(args.dir_name, meth[0]))
    for ct_idx in range(df.shape[1]):
        TS = np.array(df.iloc[:,ct_idx])
        smoothedTS = moving_avg_smoother(TS, smoother_window_length, 'center')[bfLen:-(smoother_window_length - 1 - bfLen)]
        originalData = SimAllocComb(date_value, smoothedTS, growthRateThr, pval_thr, window4Pval = window4Pval, rep_per_wk = rep_per_wk)
        originalData.growthAlarm = originalData.alarmMask_once_d_with_thr(window4Pval*rep_per_wk, originalData.computeGrowth(originalData.TimeSeries), originalData.growthRateThr, originalData.TimeSeries, ge_thr = True)
        ref_alarm = originalData.growthAlarm
        if len(ref_alarm) > 2:
            outNm = ['decentralized_noNoise', 'centralized_withNoise', 'decentralized_withNoise']
            TParr = np.zeros((len(outNm), nPs))
            FParr = np.zeros((len(outNm), nPs))
            FNarr = np.zeros((len(outNm), nPs))

            for numSites in [2, 3, 5, 8, 10, 15, 20]:
                allocShare = np.repeat(1/numSites, numSites)
                for l in range(nLoops):
                    simulatedData_noNoise = simDat(date_value, smoothedTS, allocProp = allocShare, growthRateThr = growthRateThr, pval_thr = pval_thr, window4Pval = window4Pval, rep_per_wk = rep_per_wk)
                    simulatedData_withNoise = simDat(date_value, np.random.poisson(smoothedTS), allocProp = allocShare, growthRateThr = growthRateThr, pval_thr = pval_thr, window4Pval = window4Pval, rep_per_wk = rep_per_wk)
                    for meth in methLis:
                        simulatedData_noNoise.combMethod = meth[0]
                        simulatedData_noNoise.weightMethod = meth[1]
                        simulatedData_withNoise.combMethod = meth[0]
                        simulatedData_withNoise.weightMethod = meth[1]
                        simulatedData_noNoise.get_combinePval()
                        simulatedData_withNoise.get_combinePval()
                        for pv_idx in range(len(pthr)):
                            simulatedData_noNoise.combinedPAlarm = simulatedData_noNoise.alarmMask_once_d_with_thr(window4Pval*rep_per_wk, simulatedData_noNoise.combinedP, pthr[pv_idx], originalData.TimeSeries, ge_thr = False)
                            simulatedData_withNoise.pvalAlarm = simulatedData_withNoise.alarmMask_once_d_with_thr(window4Pval*rep_per_wk, simulatedData_withNoise.pval, pthr[pv_idx], originalData.TimeSeries, ge_thr = False)
                            simulatedData_withNoise.combinedPAlarm = simulatedData_withNoise.alarmMask_once_d_with_thr(window4Pval*rep_per_wk, simulatedData_withNoise.combinedP, pthr[pv_idx], originalData.TimeSeries, ge_thr = False)
                            outIdx = 0
                            for new_alarm in [simulatedData_noNoise.combinedPAlarm, simulatedData_withNoise.pvalAlarm, simulatedData_withNoise.combinedPAlarm]:
                                TPsq_tmp, FPsq_tmp, FNsq_tmp, mappedTime_tmp = mappingAlarms(ref_alarm, new_alarm, 7, 14)
                                TParr[outIdx, pv_idx] = len(mappedTime_tmp)
                                FParr[outIdx, pv_idx] = len(FPsq_tmp)
                                FNarr[outIdx, pv_idx] = len(FNsq_tmp)
                                outIdx += 1

                        for fnm_idx, fnm in enumerate(outNm):
                            with open(os.path.join(args.dir_name, meth[0], 'TP_'+fnm+'_share'+str(numSites)+'.csv'), 'a+') as f_TP:
                                np.savetxt(f_TP, TParr[fnm_idx].reshape(1, -1), fmt='%d', delimiter=',')
                            with open(os.path.join(args.dir_name, meth[0], 'FP_'+fnm+'_share'+str(numSites)+'.csv'), 'a+') as f_FP:
                                np.savetxt(f_FP, FParr[fnm_idx].reshape(1, -1), fmt='%d', delimiter=',')
                            with open(os.path.join(args.dir_name, meth[0], 'FN_'+fnm+'_share'+str(numSites)+'.csv'), 'a+') as f_FN:
                                np.savetxt(f_FN, FNarr[fnm_idx].reshape(1, -1), fmt='%d', delimiter=',')
