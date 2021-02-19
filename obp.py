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
from numpy import linalg
import copy

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
        self.R = self.df['Close'].rolling(2).apply(lambda x: x[1]/x[0])-1    #Evaluate R_k,i = S_k,i/S_k-1,i TODO:replace Close with something better
        self.R.dropna(inplace=True)                                          #<----THIS IS UGLY
        print("start date   : {}".format(self.start_date))
        print("start date   : {}".format(self.end_date))
        print("time interval: {}".format(self.time_interval))
        return self.R.to_numpy()

def get_cov(data):
    m,n = data.shape
    cov = np.zeros(shape=(n,n))
    return np.cov(data)

def compute_stats(totlen,tau,history):    
    Sigma = []
    ExpectedReturn = []
    for k in range(floor(totlen/tau)):
        init = k*tau
        end  = init+tau
        Sigma.append(get_cov(np.transpose(history[init:end])))   #compute covariance matrix among assets
        ExpectedReturn.append(np.mean(history[init:end],axis=0))  #compute expected return for each asset
        init = end
    return Sigma, ExpectedReturn

def eigen_decomposition(steps,covariance,Nassets):
    L = []
    H = []
    for k in range(steps):
        Ltmp, Htmp = linalg.eigh(covariance[k])
        
        L.append(Ltmp[::-1])                                                 # discending order
        H.append(Htmp[:,::-1])                                               # discending order
    
        assert(np.sum([a>0 for a in L[k]]))                                  # healthy covariance check. Positive definite.
        assert(np.sum(np.transpose(H[k]).dot(H[k]))-Nassets < 10**-10)       # orthonormality check. 
        assert(np.sum(np.transpose(H[k]).dot(covariance[k].dot(H[k])) - np.diag(L[k])) < 10**-10)
    return L, H

def normalization(steps,Nassets,L,H):
    SigmaP = []
    LP = []
    for k in range(steps):
        LP.append([l/np.linalg.norm(H[k][:,n].dot(np.ones(Nassets)))**2 for n,l in enumerate(L[k])])
        for i in range(Nassets):
            H[k][:,i]=H[k][:,i]/(np.transpose(H[k][:,i]).dot(np.ones(Nassets)))
        SigmaP.append(np.transpose(H[k]).dot(Sigma[k].dot(H[k])))
        assert(np.sum([l < 10**-10 for l in LP[k] - np.diag(SigmaP[k])])-Nassets < 10**-10)
    return LP, SigmaP

def get_sharpe_ratio(steps,Nassets,H,ExpectedReturn,L):
    SR = []
    for k in range(steps):
        for i in range(Nassets):
            SR.append(H[k][:,i]*ExpectedReturn[k]/np.sqrt(L[k][i])) 
    return SR

if __name__=="__main__":
    print("#"*20)
    print(" Orthogonal Bandits Started")
    print("#"*20)
    
    ticker = ['VOW.DE','BA', 'AMD', 'AAPL']
    start_date=[2008, 9, 1]
    end_date=[2008,12,31]
    tau = 5                  #sliding window 
    
    data = Data(name=ticker)
    history   = data.get_data(
                            startdate=start_date,
                            enddate=end_date)
    totlen = len(history)
    time   = int(floor(totlen/tau))
    Nassets = len(ticker)
    print("inference time steps: {} ".format(time))
    
    Sigma,ExpectedReturn = compute_stats(
                                totlen=totlen,
                                tau=tau,
                                history=history)

    L, H = eigen_decomposition(
                        steps=time,
                        covariance=Sigma,
                        Nassets=Nassets)

    LP, SigmaP = normalization(
                        steps=time,
                        Nassets=Nassets,
                        L=L,
                        H=H)

    SR = get_sharpe_ratio(
                    steps=time,
                    Nassets=Nassets,
                    H=H,
                    ExpectedReturn=ExpectedReturn,
                    L=LP)
    print(SR)