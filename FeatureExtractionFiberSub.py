# -*- coding: utf-8 -*-
"""
Created on May 19 21:38:20 2018

@author: Diginnos
"""


import numpy as np
from scipy import fftpack
#import matplotlib.pyplot as plt
import csv

def FeatureExtractionFiberSub(file,savefile,L):
    #Fiber波形dataの処理テスト
    label=[L]
    #print(label)
    n=256             # データ数
    tws=n/2
    tws=int(tws)
    StartTime=0
    datasetNum=7168
    
    #f = 25                         # サンプリング周波数
    #dt = 1/f                       # サンプリング間隔
    #t = np.linspace(1, n, n)*dt-dt #時間分解能
    #ファイルからdataの読み込み
    datas = np.loadtxt(file, delimiter=',', dtype='float')
    roopNum=((datasetNum-np.mod(datasetNum,tws))/tws)-1
    f = open(savefile, 'a')
    writer = csv.writer(f, lineterminator='\n')  # writerオブジェクトを作成
    hammingWindow = np.hamming(n)
    #index = file.find(".csv")
    #print(file[index-2:index])

        
    for r in range(int(roopNum)):        
        #feature=[]
        #データセットの作成
        X=datas
        #切り出し
        TimeWindow=X[StartTime:StartTime+n, :]
        StartTime=StartTime+tws
        #print(r)
        #特徴計算(Mean, Energy, Entropy, Correlation)
        Mean, EnergyAve, Entropy, CC, a1, a1mu, a1sigma = [],[],[],[],[],[],[]
        for i in [0, 1, 3, 4, 5]:
            Mean.append(TimeWindow[:,i].mean())
            #FFT
            #freq = fftpack.fftfreq(n, dt)
            Amplitude = fftpack.fft(hammingWindow * TimeWindow[:,i])/(n/2)
            fil = np.copy(Amplitude)
            #Energy
            Energy = np.power(np.abs(fil[1:128]),2)#0Hzの周波数カット
            print(len(Energy))
            EnergyAve.append(sum(Energy)/len(Energy))
            
            #Entropy
            #PSD:Power Spectrum Density
            PSD=Energy/len(Energy)
            NormalizedPSD=PSD/sum(PSD)
            combined1 = [x * y for (x, y) in zip(NormalizedPSD, np.log2(NormalizedPSD))]
            Entropy.append(-sum(combined1))
       

    #writer.writerow(Mean)
    #writer.writerow(EnergyAve)
    #writer.writerow(Entropy)
        #ーーーーーーーーーーーーーーーーーーーー
        #ここまでで各データのMean、Energy、Entropyを算出
        #ーーーーーーーーーーーーーーーーーーーー

        #CC:CorrelationCoefficient
        j=0
        for i in [0, 1, 3, 4, 5]:
            a1.append(TimeWindow[:,i])#256個の1列のデータ
            a1mu.append(a1[j].mean())#平均値
            a1sigma.append(a1[j].std())#標準偏差
            j=j+1

        j=0
        k=5
        for i in range(k):
            #k=k-i
            for j in range(k):
                if i==j:
                    print(i,j,"same")
                else:
                    covariance = sum([(xi - a1mu[i]) * (yi - a1mu[j]) for xi, yi in zip(a1[i], a1[j])]) / n
                    #print('共分散:', covariance)
                    val=covariance / (a1sigma[i] * a1sigma[j])
                    CC.append(val)
                    print(i,j)

        
        feature = Mean + EnergyAve + Entropy + CC + label
        writer.writerow(feature)
    f.close()
        

    #return feature
    #print(len(CC))
    #CC=np.reshape(CC,(15, 15))
    #writer.writerows(CC)
    
    





    





