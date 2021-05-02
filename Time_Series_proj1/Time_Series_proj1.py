#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 20:50:46 2021

@author: robert
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

class finance:
    
    def __init__(self, time_series):
        
        self.time_series = np.array(time_series)
        self.avg = self.time_series.mean()
        self.var = self.time_series.var(ddof = 1)
        self.length = len(time_series)
    
    def describle(self):
        return print(f'Data平均: {self.avg}',f'Data變異數: {self.var}')
    
    def cov(self, k):
        #range(k - 1, len(self.time_series)) >> 假設k=1 data有10期 則這給了range(1,2,...10)
        #再用reversed使數列反過來為(10,9,8,....1)
        #之後用list comperhensive 將(time t與time t-k的data)與平均數的差相乘並用在list中，後續在加總並除上T-1
        cov_list = [ (self.time_series[t] - self.avg) * (self.time_series[t - k] - self.avg) for t in reversed(range(k, self.length)) ]
        cov =  sum(cov_list)/ (self.length - k - 1)
        #與np.cov(rt[:-1],rt[1::], ddof = 1)不同在於，自製的cov有weakly stationay的假設，其平均數為常數，用np.cov的話 其平均數會有些微差異，大概小數點4位
        return cov
    
    def autocorrl(self, k):
        
        return self.cov(k) / self.var



raw_data = yf.download(tickers = '2330.TW', period = '1y', interval = '1d' )
rt = []
dt = []
for t in range(1,len(raw_data['Close'])):
    Rt = 100 * (np.log(raw_data['Close'][t]) - np.log(raw_data['Close'][t - 1]))
    rt.append(Rt)
    if Rt > 0:
        dt.append(1)
    else:
        dt.append(0)


data = finance(rt)
bar_data = [data.autocorrl(k) for k in range(20)]

fig = plt.figure()
fig.suptitle('autocorrelation')
ax1 = fig.add_subplot()
ax1.set_xticks(range(21))
ax1.bar(range(21),np.array(bar_data))

random_data = finance(np.random.normal(0, 1, 1000))
ran_bar_data = [random_data.autocorrl(k) for k in range(20)]

fig = plt.figure()
fig.suptitle('autocorrelation')
ax1 = fig.add_subplot()
ax1.set_xticks(range(21))
ax1.bar(range(21),np.array(ran_bar_data))

fig = plt.figure()
ax1 = fig.add_subplot()
ax1.set_ylabel("rate of return")
ax1.set_xlabel("time")
ax1.plot(raw_data.index[1::], rt)
