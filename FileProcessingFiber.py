# -*- coding: utf-8 -*-
"""
Created on May 19 21:38:20 2018
@author: Diginnos
"""


import numpy as np
from scipy import fftpack
#import matplotlib.pyplot as plt
import csv
from matplotlib import pyplot as plt
from time import sleep
import math
import os

def file_processing(file):
    
    #Fiber取得csvファイルdataの処理
    #datas = np.loadtxt(file, delimiter=',', dtype='float')
    i=0
    calibration=[]
    X=[]
    datas=[]
    activity=[]
    act=False
    
    f=open(file)
    X=f.read().split("\n")
    f.close()

    print(len(X))
    for i in range(1,len(X)-1):
        if i==1:
            cal=X[i].split(",")
            print(cal)
            for j in range (1,9):
                #calibration=calibration(float(cal[1]))
                if float(cal[j])>0:
                    calibration.append(float(cal[j]))
                    print(cal[j])
            #print(calibration)
        elif i==2:
            print(X[i])
        else:
            cal=X[i].split(",")
            #print(cal)
            if len(cal)==1:
                activity.append(cal[0])
                print(cal[0])
                if cal[0]=='End':
                    print(len(datas))
                    datas=np.reshape(datas,(int(len(datas)/6),6))
                    #np.savetxt(activity[len(activity)-2]+'.csv', datas[400:])
                    f=open('csvdata/'+activity[len(activity)-2]+'.csv','a')
                    writer = csv.writer(f, lineterminator='\n')  # writerオブジェクトを作成
                    for j in range(400,len(datas)):
                        for m in [1,2,3,4,5]:
                            if int(datas[j,m])==0:datas[j,m]=1
                            datas[j,m]=-10*math.log10(float(datas[j,m])/float(calibration[m-1]))
                        writer.writerow(datas[j])
                    f.close()
                    #print(len(datas))
                    datas=[]                
                    act=False
                else:
                    act=True
                    
            else:
                if act:
                    datas.append(cal[0])
                    for j in [1,2,3,5,8]:
                       datas.append(int(cal[j]))
                            #print(cal[j])
    
    print(len(datas))
    print(activity)



if __name__ == '__main__':
    path='rawdata'
    
    # os.listdir('パス')
    # 指定したパス内の全てのファイルとディレクトリを要素とするリストを返す
    files = os.listdir(path)
    files_txt = [f for f in files if os.path.isfile(os.path.join(path, f))]
    print(files_txt)
    
    for i in range(len(files_txt)):
        print(file_processing(path+'/'+files_txt[i]))
