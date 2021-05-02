{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "研究金融時間序列報酬的自相關性，並用bar圖呈現\n",
    "先導入常用套件，我用了object方式寫了一些相關的功能，但其實一般function就可\n",
    "主要是為了未來可以重複引用"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import yfinance as yf\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "class finance:\n",
    "    \n",
    "    def __init__(self, time_series):\n",
    "        \n",
    "        self.time_series = np.array(time_series)\n",
    "        self.avg = self.time_series.mean()\n",
    "        self.var = self.time_series.var(ddof = 1)\n",
    "        self.length = len(time_series)\n",
    "    \n",
    "    def describle(self):\n",
    "        return print(f'Data平均: {self.avg}',f'Data變異數: {self.var}')\n",
    "    \n",
    "    def cov(self, k):\n",
    "        #k為lag的期數\n",
    "        #range(k - 1, self.length) >> 假設k=1 data有10期 則這給了range(1,2,...10)\n",
    "        #再用reversed使數列反過來為(10,9,8,....1)\n",
    "        #之後用list comperhensive 將(time t與time t-k的data)與平均數的差相乘並暫存在list中，後續在加總並除上n-k-1\n",
    "        cov_list = [ (self.time_series[t] - self.avg) * (self.time_series[t - k] - self.avg) for t in reversed(range(k, self.length)) ]\n",
    "        cov =  sum(cov_list)/ (self.length - k - 1)\n",
    "        #與np.cov(rt[:-1],rt[1::], ddof = 1)不同在於，自製的cov為weakly stationay的假設，其平均數為常數，用np.cov的話，rt[:-1]與rt[1::]其平均數會有些微差異\n",
    "        return cov\n",
    "    \n",
    "    def autocorrl(self, k):\n",
    "        \n",
    "        return self.cov(k) / self.var"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "再來我們使用yahoofianace的資料\n",
    "這次研究的個股為台積電2330，至於資料時間長度可按喜好決定\n",
    "這次示範就以3年台積電的股價做示範\n",
    "先算出其對數報酬率，並存放在rt的list當中"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[*********************100%***********************]  1 of 1 completed\n"
     ]
    }
   ],
   "source": [
    "raw_data = yf.download(tickers = '2330.TW', period = '1y', interval = '1d' )\n",
    "rt = [100 * (np.log(raw_data['Close'][t]) - np.log(raw_data['Close'][t - 1])) for t in range(1,len(raw_data['Close'])) ]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Open</th>\n",
       "      <th>High</th>\n",
       "      <th>Low</th>\n",
       "      <th>Close</th>\n",
       "      <th>Adj Close</th>\n",
       "      <th>Volume</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Date</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2020-04-29</th>\n",
       "      <td>299.0</td>\n",
       "      <td>301.5</td>\n",
       "      <td>298.0</td>\n",
       "      <td>299.0</td>\n",
       "      <td>292.370087</td>\n",
       "      <td>44059301</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2020-04-30</th>\n",
       "      <td>302.0</td>\n",
       "      <td>305.0</td>\n",
       "      <td>301.5</td>\n",
       "      <td>304.5</td>\n",
       "      <td>297.748169</td>\n",
       "      <td>55126085</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2020-05-04</th>\n",
       "      <td>294.5</td>\n",
       "      <td>296.5</td>\n",
       "      <td>294.0</td>\n",
       "      <td>295.0</td>\n",
       "      <td>288.458801</td>\n",
       "      <td>71581861</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2020-05-05</th>\n",
       "      <td>296.5</td>\n",
       "      <td>298.0</td>\n",
       "      <td>295.0</td>\n",
       "      <td>295.5</td>\n",
       "      <td>288.947723</td>\n",
       "      <td>23547405</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2020-05-06</th>\n",
       "      <td>294.5</td>\n",
       "      <td>296.0</td>\n",
       "      <td>292.5</td>\n",
       "      <td>296.0</td>\n",
       "      <td>289.436615</td>\n",
       "      <td>34240479</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "             Open   High    Low  Close   Adj Close    Volume\n",
       "Date                                                        \n",
       "2020-04-29  299.0  301.5  298.0  299.0  292.370087  44059301\n",
       "2020-04-30  302.0  305.0  301.5  304.5  297.748169  55126085\n",
       "2020-05-04  294.5  296.5  294.0  295.0  288.458801  71581861\n",
       "2020-05-05  296.5  298.0  295.0  295.5  288.947723  23547405\n",
       "2020-05-06  294.5  296.0  292.5  296.0  289.436615  34240479"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "raw_data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Open</th>\n",
       "      <th>High</th>\n",
       "      <th>Low</th>\n",
       "      <th>Close</th>\n",
       "      <th>Adj Close</th>\n",
       "      <th>Volume</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Date</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2021-04-23</th>\n",
       "      <td>592.0</td>\n",
       "      <td>602.0</td>\n",
       "      <td>590.0</td>\n",
       "      <td>602.0</td>\n",
       "      <td>602.0</td>\n",
       "      <td>27754511</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-04-26</th>\n",
       "      <td>605.0</td>\n",
       "      <td>610.0</td>\n",
       "      <td>603.0</td>\n",
       "      <td>610.0</td>\n",
       "      <td>610.0</td>\n",
       "      <td>30887664</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-04-27</th>\n",
       "      <td>608.0</td>\n",
       "      <td>611.0</td>\n",
       "      <td>605.0</td>\n",
       "      <td>610.0</td>\n",
       "      <td>610.0</td>\n",
       "      <td>26317481</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-04-28</th>\n",
       "      <td>606.0</td>\n",
       "      <td>608.0</td>\n",
       "      <td>601.0</td>\n",
       "      <td>602.0</td>\n",
       "      <td>602.0</td>\n",
       "      <td>24024054</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-04-29</th>\n",
       "      <td>609.0</td>\n",
       "      <td>609.0</td>\n",
       "      <td>600.0</td>\n",
       "      <td>600.0</td>\n",
       "      <td>600.0</td>\n",
       "      <td>31828333</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "             Open   High    Low  Close  Adj Close    Volume\n",
       "Date                                                       \n",
       "2021-04-23  592.0  602.0  590.0  602.0      602.0  27754511\n",
       "2021-04-26  605.0  610.0  603.0  610.0      610.0  30887664\n",
       "2021-04-27  608.0  611.0  605.0  610.0      610.0  26317481\n",
       "2021-04-28  606.0  608.0  601.0  602.0      602.0  24024054\n",
       "2021-04-29  609.0  609.0  600.0  600.0      600.0  31828333"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "raw_data.tail()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[1.8441427902722296,\n",
       " 1.3201511858536463,\n",
       " 0.0,\n",
       " -1.3201511858536463,\n",
       " -0.33277900926744763]"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "rt[-5::] #台積電近五天的對數報酬率"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "再來我們就可以使用寫好的object來計算lag不同其數台積電對數報酬率的自相關係數了，並用matplotlab繪製bar圖\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [],
   "source": [
    "#研究20期\n",
    "data = finance(rt)\n",
    "bar_data = [data.autocorrl(k) for k in range(21)]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<BarContainer object of 21 artists>"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig = plt.figure()\n",
    "fig.suptitle('autocorrelation')\n",
    "ax1 = fig.add_subplot()\n",
    "ax1.set_xticks(range(21))\n",
    "ax1.bar(range(21),np.array(bar_data))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "我們用常態分配隨機創造出來的white noise來比較看看"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<BarContainer object of 21 artists>"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "random_data = finance(np.random.normal(0, 1, 1000))\n",
    "ran_bar_data = [random_data.autocorrl(k) for k in range(21)]\n",
    "\n",
    "fig = plt.figure()\n",
    "fig.suptitle('autocorrelation')\n",
    "ax1 = fig.add_subplot()\n",
    "ax1.set_xticks(range(21))\n",
    "ax1.bar(range(21),np.array(ran_bar_data))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "可見相較於白噪音完全隨機的自相關係數\n",
    "台積電的股價報酬明顯有自我相關\n",
    "然而台積電的股價報酬嚴格來說並不符合時間序列中的平穩假設\n",
    "尤其是變異數為常數的假設\n",
    "我們可以藉由plot台積電的報酬率出來"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x7fa2d4dcd190>]"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig = plt.figure()\n",
    "ax1 = fig.add_subplot()\n",
    "ax1.set_ylabel(\"rate of return\")\n",
    "ax1.set_xlabel(\"time\")\n",
    "ax1.plot(raw_data.index[1::], rt)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "明顯在2020/7-2020/9間報酬率的波動大\n",
    "但在2020/11-2021/1月間報酬率的波動小\n",
    "之後的更新也會注重在檢定時間序列的平穩性與一些模型的回測上\n",
    "感謝收看！\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
