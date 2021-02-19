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
        self.R = self.df['Close'].rolling(2).apply(lambda x: x[1]/x[0])-1    #Evaluate R_k,i = S_k,i/S_k-1,i
        self.R.dropna(inplace=True)                                          #<----THIS IS UGLY
        print("start date   : {}".format(self.start_date))
        print("start date   : {}".format(self.end_date))
        print("time interval: {}".format(self.time_interval))
        return self.R.to_numpy()

def get_cov(data):
    m,n = data.shape
    cov = np.zeros(shape=(n,n))
    return np.cov(data)
    
if __name__=="__main__":
    print("#"*20)
    print(" Orthogonal Bandits Started")
    print("#"*20)
    
    ticker = ['VOW.DE','BA', 'AMD', 'AAPL']
    start_date=[2008, 9, 1]
    end_date=[2008,12,31]
    tau = 5             #sliding window 
    
    data = Data(name=ticker)
    R   = data.get_data(startdate=start_date,enddate=end_date)
    totlen = len(R)
    Sigma = []
    for i in range(floor(totlen/tau)):
        init = i*tau
        end  = init+tau
        Sigma.append(get_cov(R[init:end]))
        init = end
    print(Sigma[0])
