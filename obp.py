#David Preti
from pandas_datareader import data as pdr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import random 
import argparse 
from tqdm import tqdm
from math import floor


class Data():
    def __init__(self,name):
        self.name = name
        print("Asset        : {}".format(self.name))

    def get_data(self,startdate,enddate):
        #input date format [2008,9,1]
        self.start_date=datetime.datetime(startdate[0],startdate[1],startdate[2])
        self.end_date=datetime.datetime(enddate[0],enddate[1],enddate[2])
        self.time_interval=self.end_date-self.start_date
        self.df = pdr.get_data_yahoo(self.name, start=self.start_date, end=self.end_date)
        self.df.drop(["Adj Close"],axis=1,inplace=True)
        self.R = self.df['Close'].rolling(2).apply(lambda x: x[1]/x[0])      #Evaluate R_k,i = S_k,i/S_k-1,i TODO:replace Close with something better
        self.R.dropna(inplace=True)                                          #<----THIS IS UGLY
        print("start date   : {}".format(self.start_date))
        print("start date   : {}".format(self.end_date))
        print("time interval: {}".format(self.time_interval))
        return self.R.to_numpy()

def get_cov(data):
    m,n = data.shape
    cov = np.zeros(shape=(n,n))
    return np.cov(data)

def compute_stats(totlen,tau,history,k):    
    Sigma = []
    ExpectedReturn = []
    init = k
    end  = k+tau                         # NOTICE that: inference time t=k+tau is not included!
    Sigma = get_cov(np.transpose(history[init:end]))    #compute covariance matrix among assets
    ExpectedReturn = np.mean(history[init:end],axis=0)  #compute expected return for each asset
    init = end
    return Sigma, ExpectedReturn

def eigen_decomposition(covariance,Nassets):
    Ltmp, Htmp = np.linalg.eigh(covariance)
    L = Ltmp[::-1]                                                 # discending order
    H = Htmp[:,::-1]                                               # discending order
    #assert(np.sum([a>0 for a in L])-len(L)==0)                     # healthy covariance check. Positive definite.
    assert(np.sum(np.transpose(H).dot(H))-Nassets < 10**-10)       # orthonormality check. 
    assert(np.sum(np.transpose(H).dot(covariance.dot(H)) - np.diag(L)) < 10**-10)
    return L, H

def normalization(Nassets,L,H):
    LP = [l/np.linalg.norm(H[:,n].dot(np.ones(Nassets)))**2 for n,l in enumerate(L)]
    for i in range(Nassets):
        H[:,i]=H[:,i]/(np.transpose(H[:,i]).dot(np.ones(Nassets)))
    SigmaP = np.transpose(H).dot(Sigma.dot(H))
    assert(np.sum([l < 10**-10 for l in LP - np.diag(SigmaP)])-Nassets < 10**-10)
    return LP, H

def get_sharpe_ratio(Nassets,H,ExpectedReturn,L):
    sr = []
    for i in range(Nassets):
        sr.append(np.sum(H[:,i]*ExpectedReturn)/np.sqrt(np.abs(L[i]))) #abs here is introduced for badnly behaved eivenvalues
    return sr

def ucb(R,t,N,tau):
    ca = float("inf")
    d = [np.sqrt(float(2*np.log(t+tau)/(tau+N[i]))) if (tau+N[i])!=0 else ca for i in range(len(N))]
    return np.random.choice(np.flatnonzero((R+d)==np.max(R+d)))
            
def split_assets(Nsign,sr,L,H):
    sr_sign = np.array(sr[:Nsign])
    sr_insign = np.array(sr[Nsign:])
    L_sign = L[:Nsign]
    L_insign = L[Nsign:]
    H_sign = H[:,:Nsign]
    H_insign = H[:,Nsign:]
    return sr_sign,sr_insign,L_sign,L_insign,H_sign,H_insign

if __name__=="__main__":
    print("#"*20)
    print(" Orthogonal Bandits Started")
    print("#"*20)
    
    ticker = ['VOW.DE','BA', 'AMD', 'AAPL','GME','CVGW','CAMP','WSCI','LNDC','WOR']
    start_date=[2007, 9, 1]
    end_date=[2020,1,9]
    tau = 500                  #sliding window 
    Nsign = 3                  #Significative portfolios 
    data = Data(name=ticker)
    history   = data.get_data(
                            startdate=start_date,
                            enddate=end_date)
    totlen = len(history)
    print(' Effective available steps: {}'.format(totlen))
    time   = int(floor(totlen-tau-1))
    Nassets = len(ticker)
    print("inference time steps: {} ".format(time))
    
    N = [np.zeros(Nsign), np.zeros(Nassets-Nsign)]
    mu = []
    mu_nss = []
    muEW = []
    weights = []
    weights_NSS = []
    theta_hist = []
    for k in range(time):
        Sigma,ExpectedReturn = compute_stats(
                                    totlen=totlen,
                                    tau=tau,
                                    history=history,
                                    k=k)

        L, H = eigen_decomposition(
                            covariance=Sigma,
                            Nassets=Nassets)

        LP, HP = normalization(
                            Nassets=Nassets,
                            L=L,
                            H=H)

        sr = get_sharpe_ratio(
                        Nassets=Nassets,
                        H=H,
                        ExpectedReturn=ExpectedReturn,
                        L=L)

        sr_sign, sr_insign, LP_sign, LP_insign, H_sign, H_insign = split_assets(Nsign=Nsign, sr=sr, L=LP, H=HP)
        best_assets = [0,0]
        sets = [sr_sign, sr_insign]
        for ss,s in enumerate(sets):
            ii = ucb(R=s,t=k,N=N[ss],tau=tau)
            N[ss][ii]+=1
            best_assets[ss]=ii

        theta = LP_sign[best_assets[0]]/(LP_sign[best_assets[0]] + LP_insign[best_assets[1]])
        w = (1-theta)*H_sign[:,best_assets[0]] + theta*H_insign[:,best_assets[1]]
        weights.append(w)
        assert(np.sum(w)-1<10**-10)

        #NO SHORT SELL ---
        w_nss = [a if a>0 else 0 for a in w]
        w_nss = w_nss/np.sum(w_nss)
        weights_NSS.append(w_nss)
        assert(np.sum(w_nss)-1<10**-10)
        # ---- 

        kp1 = tau + k
        muk = np.sum(w*history[kp1])-1
        muk_nss = np.sum(w_nss*history[kp1])-1
        mukEW = np.sum(history[kp1]/Nassets)-1
        mu.append(muk)
        mu_nss.append(muk_nss)
        muEW.append(mukEW)  
        theta_hist.append(theta)
 
    Emu=np.mean(mu)
    Smu=np.std(mu)
    print('Mean Sharpe Ratio: {}({}) normalized SR {}'.format(Emu,Smu,Emu/Smu))
    #cw = np.prod(np.array([m+1 for m in mu]))
    #cwEW = np.prod(np.array([m+1 for m in muEW]))
    #print('Cumulative Reward EW : {}'.format(cw))
    #print('Cumulative Reward OBL: {}'.format(cwEW))
    

    ccw = 1
    history_cw = []
    for k in range(time):
        ccw *=(mu[k]+1)
        history_cw.append(ccw)

    ccw_nss = 1
    history_cw_nss = []
    for k in range(time):
        ccw_nss *=(mu_nss[k]+1)
        history_cw_nss.append(ccw_nss)

    ccwEW = 1
    history_cwEW = []
    for k in range(time):
        ccwEW *=(muEW[k]+1)
        history_cwEW.append(ccwEW)
    

    fig = plt.figure()
    plt.plot(history_cw)
    plt.plot(history_cw_nss)
    plt.plot(history_cwEW)
    plt.legend(['OBL','OBL-no-shortsale','EW'])
    plt.xlabel('time')
    plt.ylabel('Comulative Reward')
    plt.axhline(y=1, color='r', linestyle='--')
    plt.savefig('./fig/OBPvsEW.png')

    fig = plt.figure()
    plt.plot(theta_hist)
    plt.xlabel('time')
    plt.ylabel('theta mixing')
    plt.axhline(y=1, color='r', linestyle='--')
    plt.savefig('./fig/theta.png')
    
    fig = plt.figure()
    plt.plot(np.dot(100,weights[:]))
    plt.xlabel('time')
    plt.ylabel('invested % of wealth')
    plt.axhline(y=100/Nassets, color='b', linestyle=':')
    plt.legend(ticker+['EW strategy'])
    plt.axhline(y=0, color='r', linestyle='--')
    plt.axhline(y=100, color='g', linestyle='--') 
    plt.savefig('./fig/weights.png')

    fig = plt.figure()
    plt.plot(np.dot(100,weights_NSS[:]))
    plt.xlabel('time')
    plt.ylabel('invested % of wealth')
    plt.axhline(y=100/Nassets, color='b', linestyle=':')
    plt.legend(ticker+['EW strategy'])
    plt.axhline(y=0, color='r', linestyle='--')
    plt.axhline(y=100, color='g', linestyle='--') 
    plt.savefig('./fig/weights_nss.png')
    
    