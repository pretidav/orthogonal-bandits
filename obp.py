#David Preti
from pandas_datareader import data as pdr
import pandas as pd
import numpy as np
import datetime

class Data():
    def __init__(self,name):
        self.name = name
        print("Asset        : {}".format(self.name))
    def get_data(self,startdate,enddate):
        #date in format [2008,9,1]
        self.start_date=datetime.datetime(startdate[0],startdate[1],startdate[2])
        self.end_date=datetime.datetime(enddate[0],enddate[1],enddate[2])
        self.time_interval=self.end_date-self.start_date
        self.df = pdr.get_data_yahoo(self.name, start=self.start_date, end=self.end_date)
        self.df.drop("Adj Close",axis=1,inplace=True)
        print("start date   : {}".format(self.start_date))
        print("start date   : {}".format(self.end_date))
        print("time interval: {}".format(self.time_interval))
        return self.df

if __name__=="__main__":
    print("#"*20)
    print(" Orthogonal Bandits Started")
    print("#"*20)
    
    ticker = 'VOW.DE'
    start_date=[2008, 9, 1]
    end_date=[2008,12,31]
    
    data = Data(name=ticker)
    df   = data.get_data(startdate=start_date,enddate=end_date) 
    print(df.columns)
