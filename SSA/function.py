import numpy as np
import keras
from keras.models import load_model
# import tensorflow as tf
from matplotlib.pyplot import *

model=load_model('gee3_hole11_position_model.h5', compile=False)

generator=load_model('cgan_gee3_2D_hole11_modelcgan_generator.h5', compile=False)


class funtion():
    def __init__(self):
        print("starting SSA")
def Parameters(F):
    if F=='F1':
        # ParaValue=[-100,100,30] [-100,100]代表初始范围，30代表dim维度
        ParaValue = [-100, 100, 30]

    elif F=='F2':
        ParaValue = [-10, 10, 30]

    elif F=='F3':
        ParaValue = [-100, 100, 30]

    elif F=='F4':
        ParaValue = [-100, 100, 30]

    elif F=='F5':
        ParaValue = [-30, 30, 30]

    elif F=='F6':
        ParaValue = [-100, 100, 30]
    elif F=='F7':
        ParaValue = [-10, 10, 4]
    elif F=='F8':
        # ParaValue = [-10, 10, 5]
        ParaValue = [-10, 10, 5]
    elif F=='F9':
        ParaValue = [-10, 10, 4]

    elif F=='F10':#trench
        ParaValue = [-10, 10, 8]
        
    elif F=='F11':#GEE3
        ParaValue = [0, 1, 17]
    return ParaValue
# 标准测试函数采用单峰测试函数（Dim = 30），计算适应度
def fun(F,X):  # F代表函数名，X代表数据列表
    if F == 'F1':
        O = np.sum(X*X)

    elif F == 'F2':
        O = np.sum(np.abs(X))+np.prod(np.abs(X))

    elif F == 'F3':
        O = 0
        for i in range(len(X)):
            O = O+np.square(np.sum(X[0:i+1]))


    elif F == 'F4':
        O = np.max(np.abs(X))

    elif F=='F5':
        X_len = len(X)
        O = np.sum(100 * np.square(X[1:X_len] - np.square(X[0:X_len - 1]))) + np.sum(np.square(X[0:X_len - 1] - 1))

    elif F == 'F6':
        O = np.sum(np.square(np.abs(X+0.5)))

    elif F == 'F7':
        # O = np.sum(np.square(np.abs(X+0.5)))


        x = []
        y = []
        # x = [10, -10, 10, 10]
        y = np.zeros((1, 4))

        y[0][0] = X[0] / 5 + 12.5
        y[0][1] = X[1] * 2 + 35
        y[0][2] = X[2] / 100 + 1.2
        y[0][3] = X[3] / 20 + 1

        Y_pred = model.predict(y)
        # print(np.shape(y))
        # print(y)
        # print(np.shape(Y_pred))

        pred = np.zeros((64, 256))
        # for g in range(0,23):
        for i in range(0, 64):
            for j in range(0, 256):
                pred[i][j] = Y_pred[0][i][j][0]
        pred0 = pred.flatten()
        # print(np.shape(pred0))
        O = 0
        for i in range(0, 16384):
            O = O + pred0[i]
        O = O / 16384
        print(O)
        print(X)
    elif F == 'F8':
        # O = np.sum(np.square(np.abs(X+0.5)))


        x = []
        y = []
        # x = [10, -10, 10, 10]
        y = np.zeros((1, 5))

        y[0][0] = X[0] *4.5 + 45
        y[0][1] = X[1] * 3 + 30
        y[0][2] = X[2] *0.05 + 2
        y[0][3] = X[3] / 20 + 1
        y[0][4] = X[4] / 20 + 1
        # y[0][4] = 0.5
        Y_pred = model.predict(y)
        # print(np.shape(y))
        # print(y)
        # print(np.shape(Y_pred))

        pred = np.zeros((128, 128))
        # for g in range(0,23):
        for i in range(0, 128):
            for j in range(0, 128):
                pred[i][j] = Y_pred[0][i][j][0]
        pred0 = pred.flatten()
        # print(np.shape(pred0))
        O = 0
        for i in range(0, 16384):
            O = O + pred0[i]
        O = O / 16384
        print(O)
        print(X)
    elif F == 'F9':
        x = []
        y = []
        # x = [10, -10, 10, 10]
        y = np.zeros((1, 4))

        y[0][0] = (X[0] / 5 + 2)/4
        y[0][1] = (X[1] * 2 + 20)/40
        y[0][2] = (X[2] / 100 + 0.1)/0.2
        y[0][3] = X[3] / 20 + 0.5

        Y_pred = model0.predict(y)
        # print(np.shape(y))
        # print(y)
        # print(np.shape(Y_pred))

        pred = np.zeros(100)
        # for g in range(0,23):
        for i in range(0, 100):
            pred[i] = Y_pred[0][i]
        pred0 = pred.flatten()
        # print(np.shape(pred0))
        O = 0
        for i in range(0, 100):
            O = O + pred0[i]
        O = O / 100
        print(O)
        print(X)
        # O = np.sum(np.square(np.abs(X+0.5)))

    elif F == 'F10':
        # O = np.sum(np.square(np.abs(X+0.5)))


        x = []
        y = []
        # x = [10, -10, 10, 10]
        c = np.zeros((1, 8))
        k = np.zeros((1, 8))

        c[0][0] = (X[0] *0.025 + 0.25)/0.5#depth
        c[0][1] = X[1] * 0.13 + 3.7#width
        c[0][2] = (X[2] *1.5 + 15)/30#compound angle
        c[0][3] = X[3] *0.5 + 15
        c[0][4] = X[4] *0.5 + 15
        c[0][5] = X[5] *0.5 + 15
        c[0][6] = X[6] *0.5 + 15
        c[0][7] = X[7] *0.05 + 1-0.5 #blowing ratio

        k[0][0] = (X[0]) / 0.5  # depth
        k[0][1] = X[1]  # width
        k[0][2] = X[2] / 30  # compound angle
        k[0][3] = c[0][3]
        k[0][4] = c[0][3] + c[0][4]
        k[0][5] = c[0][3] + c[0][4] + c[0][5]
        k[0][6] = c[0][3] + c[0][4] + c[0][5] + c[0][6]
        k[0][7] = X[7] - 0.5  # blowing ratio

        c[0][0] = c[0][0]  # depth
        c[0][1] = c[0][1]  # width
        c[0][2] = c[0][2]  # compound angle
        c[0][3] = k[0][3]
        c[0][4] = k[0][4]
        c[0][5] = k[0][5]
        c[0][6] = k[0][6]
        c[0][7] = c[0][7]  # blowing ratio

        # y[0][0] = X[0] * 0.015 + 0.35  # depth
        # y[0][1] = X[1] * 0.04 + 2.8  # width
        # y[0][2] = X[2] * 1.5 + 15  # compound angle
        # y[0][3] = X[3] * 0.5 + 15
        # y[0][4] = X[4] * 0.5 + 15
        # y[0][5] = X[5] * 0.5 + 15
        # y[0][6] = X[6] * 0.5 + 15
        # y[0][7] = X[4] * 0.05 + 1 # blowing ratio


        j=0;
        b=np.zeros((1,512,3))
        for i in range(512):
            #         low=c[j][3]-c[j][1]/2
            #         high=c[j][3]+c[j][1]/2
            b[j][i][2] = c[j][7]

            if i * 120 / 512 >= c[j][3] - c[j][1] / 2 and i * 120 / 512 <= c[j][3] + c[j][1] / 2:
                b[j][i][0] = c[j][0]
                b[j][i][1] = c[j][2]
                b[j][i][1] = c[j][2]
            elif i * 120 / 512 >= c[j][4] - c[j][1] / 2 and i * 120 / 512 <= c[j][4] + c[j][1] / 2:
                b[j][i][0] = c[j][0]
                b[j][i][1] = c[j][2]
                b[j][i][1] = c[j][2]
            elif i * 120 / 512 >= c[j][5] - c[j][1] / 2 and i * 120 / 512 <= c[j][5] + c[j][1] / 2:
                b[j][i][0] = c[j][0]
                b[j][i][1] = c[j][2]
                b[j][i][1] = c[j][2]
            elif i * 120 / 512 >= c[j][6] - c[j][1] / 2 and i * 120 / 512 <= c[j][6] + c[j][1] / 2:
                b[j][i][0] = c[j][0]
                b[j][i][1] = c[j][2]
                b[j][i][1] = c[j][2]


        # y[0][4] = 0.5
        Y_pred = model.predict(b)
        # print(np.shape(y))
        # print(y)
        # print(np.shape(Y_pred))

        pred = np.zeros((1, 512))
        # for g in range(0,23):
        for j in range(0, 512):
            pred[0][j] = Y_pred[0][j]

        pred0 = pred.flatten()
        # print(np.shape(pred0))
        O = 0
        for i in range(0, 512):
            O = O + pred0[i]
        O = -O / 512
        print(O)
        print(X)

        if O<-0.46:
            f = open("out.txt", "w")  # 打开文件以便写入
            print(O, file=f)
            print(X,  file=f)
            f.close  # 关闭文件

    
    
    elif F == 'F11':
    #GEE3
        # O = np.sum(np.square(np.abs(X+0.5)))


        x = []
        y = []
        # x = [10, -10, 10, 10]
        c = np.zeros((1, 17))
        
        print(X)
        input=np.zeros((1,17))
        
        for i in range(17):
            input[0][i]=X[i]
            
        #constrain
        input[0][11]=X[11]*0.8+0.2
        input[0][14]=X[14]*0.8+0.2
        output = model.predict(input)
        
        
        
        for i in range(11):
            output[0][i]=round(output[0][i]*256)
            
        gru_input = np.zeros((1,256,3)) 

        for i in range(1):
            a = int(output[i][0])
            aa = int(14-2*j)
            bb = int(14-2*j+1)
        #         print(bb)
            gru_input[i][a][0]=input[i][7]#diameter
            gru_input[i][a][1]=input[i][15]#incline angle
            gru_input[i][a][2]=input[i][16]#compound angle
            a = int(output[i][1])
        #         print(bb)
            gru_input[i][a][0]=input[i][6]#diameter
            gru_input[i][a][1]=input[i][14]#incline angle
            gru_input[i][a][2]=input[i][16]#compound angle    
            a = int(output[i][2])
        #         print(bb)
            gru_input[i][a][0]=input[i][5]#diameter
            gru_input[i][a][1]=input[i][13]#incline angle
            gru_input[i][a][2]=input[i][16]#compound angle  
            a = int(output[i][3])
        #         print(bb)
            gru_input[i][a][0]=input[i][4]#diameter
            gru_input[i][a][1]=input[i][12]#incline angle
            gru_input[i][a][2]=input[i][16]#compound angle  
            a = int(output[i][4])
        #         print(bb)
            gru_input[i][a][0]=input[i][3]#diameter
            gru_input[i][a][1]=input[i][11]#incline angle
            gru_input[i][a][2]=input[i][16]#compound angle  
            a = int(output[i][5])
        #         print(bb)
            gru_input[i][a][0]=input[i][2]#diameter
            gru_input[i][a][1]=input[i][10]#incline angle
            gru_input[i][a][2]=input[i][16]#compound angle  
            a = int(output[i][6])
        #         print(bb)
            gru_input[i][a][0]=input[i][1]#diameter
            gru_input[i][a][1]=input[i][9]#incline angle
            gru_input[i][a][2]=input[i][16]#compound angle  
            a = int(output[i][7])
        #         print(bb)
            gru_input[i][a][0]=input[i][0]#diameter
            gru_input[i][a][1]=input[i][8]#incline angle
            gru_input[i][a][2]=input[i][16]#compound angle  




            a = int(output[i][8])
            gru_input[i][a][0]=(0.305-0.2)/(0.45-0.2)
            gru_input[i][a][1]=1
            a = int(output[i][9])
            gru_input[i][a][0]=(0.24-0.2)/(0.45-0.2)
            gru_input[i][a][1]=1
            a = int(output[i][10])
            gru_input[i][a][0]=(0.305-0.2)/(0.45-0.2)
            gru_input[i][a][1]=1

        pred=generator.predict(gru_input)
        O=0
        for i in range(256):
            for j in range(256):
                O=O+pred[0][i][j]
        O=0-O/(256*256)
        


        print(O)
        print(X)

        if O<-0.46:
            f = open("out.txt", "w")  
            print(O, file=f)
            print(X,  file=f)
            f.close  

    return O


def Bounds(s,Lb,Ub):
    temp = s
    for i in range(len(s)):
        if temp[i]<Lb[0,i]:
            temp[i]=Lb[0,i]
        elif temp[i]>Ub[0,i]:
            temp[i]=Ub[0,i]

    return temp


def SSA(pop,M,c,d,dim,f):
    #global fit
    P_percent=0.2
    pNum = round(pop*P_percent)  
    lb = c*np.ones((1,dim))  
    ub = d*np.ones((1,dim))  
    X = np.zeros((pop,dim))  
    fit = np.zeros((pop,1))   

    for i in range(pop):
        X[i,:] = lb+(ub-lb)*np.random.rand(1,dim)  
        fit[i,0] = fun(f,X[i,:])  


    pFit = fit  
    pX = X  
    fMin = np.min(fit[:,0]) 
    bestI = np.argmin(fit[:,0])
    bestX = X[bestI,:]
    Convergence_curve = np.zeros((1,M))  
    for t in range(M):
        sortIndex = np.argsort(pFit.T) 
        fmax = np.max(pFit[:,0])  
        B = np.argmax(pFit[:,0])  
        worse = X[B,:]  

        r2 = np.random.rand(1) 
        
        if r2 < 0.8: 
            for i in range(pNum):
                r1 = np.random.rand(1)
                X[sortIndex[0,i],:] = pX[sortIndex[0,i],:]*np.exp(-(i)/(r1*M))  
                X[sortIndex[0,i],:] = Bounds(X[sortIndex[0,i],:],lb,ub)  
                fit[sortIndex[0,i],0] = fun(f,X[sortIndex[0,i],:])  
        elif r2 >= 0.8: 
            for i in range(pNum):
                Q = np.random.rand(1)  
                X[sortIndex[0,i],:] = pX[sortIndex[0,i],:]+Q*np.ones((1,dim)) 
                X[sortIndex[0,i],:] = Bounds(X[sortIndex[0,i],:],lb,ub)
                fit[sortIndex[0,i],0] = fun(f,X[sortIndex[0,i],:])
        bestII = np.argmin(fit[:,0])
        bestXX = X[bestII,:]


        for ii in range(pop-pNum):
            i = ii+pNum
            A = np.floor(np.random.rand(1,dim)*2)*2-1
            if i > pop/2:  
                Q = np.random.rand(1)  
                X[sortIndex[0,i],:] = Q*np.exp(worse-pX[sortIndex[0,i],:]/np.square(i))
            else:  
                X[sortIndex[0,i],:] = bestXX+np.dot(np.abs(pX[sortIndex[0,i],:]-bestXX),1/(A.T*np.dot(A,A.T)))*np.ones((1,dim))
            X[sortIndex[0,i],:] = Bounds(X[sortIndex[0,i],:],lb,ub)
            fit[sortIndex[0,i],0] = fun(f,X[sortIndex[0,i],:])


        arrc = np.arange(len(sortIndex[0,:]))

        c = np.random.permutation(arrc)  
        b = sortIndex[0,c[0:20]]
        for j in range(len(b)):
            if pFit[sortIndex[0,b[j]],0] > fMin:
                X[sortIndex[0,b[j]],:] = bestX+np.random.rand(1,dim)*np.abs(pX[sortIndex[0,b[j]],:]-bestX)
            else:
                X[sortIndex[0,b[j]],:] = pX[sortIndex[0,b[j]],:]+(2*np.random.rand(1)-1)*np.abs(pX[sortIndex[0,b[j]],:]-worse)/(pFit[sortIndex[0,b[j]]]-fmax+10**(-50))
            X[sortIndex[0,b[j]],:] = Bounds(X[sortIndex[0,b[j]],:],lb,ub)
            fit[sortIndex[0,b[j]],0] = fun(f,X[sortIndex[0,b[j]]])
        for i in range(pop):

            if fit[i,0] < pFit[i,0]:
                pFit[i,0] = fit[i,0]
                pX[i,:] = X[i,:]
            if pFit[i,0] < fMin:
                fMin = pFit[i,0]
                bestX = pX[i,:]
        Convergence_curve[0,t] = fMin
        #print(fMin)
        #print(bestX)
    return fMin, bestX, Convergence_curve






