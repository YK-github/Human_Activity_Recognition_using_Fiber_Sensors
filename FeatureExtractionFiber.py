# -*- coding: utf-8 -*-
"""
Created on May 19 21:38:20 2018

@author: Diginnos
"""

import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
import csv

#import datetime


#加速度波形dataの処理テスト

n=256             # 1セットあたりのデータ数:(256/53)秒分のデータ
tws=n/2
StartTime=0

f = 53           # サンプリング周波数：何Hzで計測したか                    
dt = 1/f         # サンプリング間隔
t = np.linspace(1, n, n)*dt-dt #時間分解能
datasetNum=7168

#ファイルからdataの読み込み
datas = np.loadtxt('20180519_a1.csv', delimiter=',', dtype='float')
roopNum=((datasetNum-np.mod(datasetNum,tws))/tws)-1
f = open('Feature.csv', 'a')
writer = csv.writer(f, lineterminator='\n')  # writerオブジェクトを作成

for r in range(int(roopNum)): 
    #データセットの作成
    X=datas
    #切り出し
    TimeWindow=X[StartTime:StartTime+n, :]
    #print(TimeWindow)
    StartlTime=StartTime+tws
    
    #print(X)
    
    #特徴計算(Mean, Energy, Enntropy, Correlation)
    
    #RightElbow=TimeWindow[:,0]
    #LeftElbow=TimeWindow[:,1]
    #Back=TimeWindow[:,2]
    #LeftLeg=TimeWindow[:,3]
    #Hip=TimeWindow[:,4]
    #RightLeg=TimeWindow[:,5]
    
    Mean, EnergyAve, Entropy, CC, a1, a1mu, a1sigma = [],[],[],[],[],[],[]
    for i in [0, 1, 3, 4, 5]:
        Mean.append(TimeWindow[:,i].mean())
        #FFT
        #freq = fftpack.fftfreq(n, dt)
        Amplitude = fftpack.fft(TimeWindow[:,i])/(n/2)
        fil = np.copy(Amplitude)
        #Energy
        Energy = np.power(np.abs(fil[0:128]),2)
        EnergyAve.append(sum(Energy)/len(Energy))
        #print(len(Energy))
        
        #Entropy
        #PSD:Power Spectrum Density
        PSD=Energy/len(Energy)
        NormalizedPSD=PSD/sum(PSD)
        #print(NormalizedPSD)
        combined1 = [x * y for (x, y) in zip(NormalizedPSD, np.log2(NormalizedPSD))]
        #print(combined1)
        Entropy.append(-sum(combined1))
        """
        header=['RightElbow', 'LeftElbow', 'LeftLeg',
            'Hip', 'RightLeg']
        """

        #ーーーーーーーーーーーーーーーーーーーー
        #ここまでで各データのMean、Energy、Entropyを算出
        #ーーーーーーーーーーーーーーーーーーーー
  
    #CC:CorrelationCoefficient
    j=0
    for i in [0, 1, 3, 4, 5]:
        a1.append(TimeWindow[:,i])
        a1mu.append(a1[j].mean())
        a1sigma.append(a1[j].std())
        j=j+1

   
    k=5
    m=0
    for i in range(k):
        k=k-i
        for j in range(k):
            covariance = sum([(xi - a1mu[i]) * (yi - a1mu[j+m]) for xi, yi in zip(a1[i], a1[j+m])]) / n
            #print('共分散:', covariance)
            val=covariance / (a1sigma[i] * a1sigma[j+m])
            CC.append(val)
        m=m+1

    
    feature = Mean + EnergyAve + Entropy + CC
    writer.writerow(feature)
    
f.close()

#plt.xlim(0, 2)#y軸の範囲の指定





    





