# -*- coding: utf-8 -*-
'''
โค้ดจากบทความในบล็อก หน้า
https://phyblas.hinaboshi.com/20161228
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

def sigmoid(x):
    return 1/(1+np.exp(-x))

class ThotthoiLogistic:
    def __init__(self,eta):
        self.eta = eta

    def rianru(self,X,z,n_thamsam,n_batch=0):
        n = len(z) # จำนวนข้อมูลทั้งหมด
        # ถ้าไม่ได้กำหนดจำนวนแบตช์ หรือจำนวนแบตช์มากกว่าจำนวนข้อมูล
        if(n_batch==0 or n<n_batch):
            n_batch = n # ให้ทำด้วยจำนวนทั้งหมด (คือไม่ทำมินิแบตช์)
        X_std = X.std()
        X_mean = X.mean()
        X = (X-X_mean)/X_std # ทำให้เป็นมาตรฐาน
        self.w = np.zeros(X.shape[1]+1)
        self.entropy = []
        self.thuktong = []
        for j in range(n_thamsam):
            # สุ่มเลขลำดับการเลือก
            lueak = np.random.permutation(n)
            for i in range(0,n,n_batch):
                # เลือก X และ z บางส่วนตามลำดับ
                Xn = X[lueak[i:i+n_batch]]
                zn = z[lueak[i:i+n_batch]]
                # ปรับค่าน้ำหนัก
                phi = self.ha_sigmoid(Xn)
                eee = (zn-phi)/len(zn)*self.eta
                self.w[1:] += np.dot(eee,Xn)
                self.w[0] += eee.sum()
            # คำนวณและบันทึกผลในแต่ละรอบ
            thukmai = self.thamnai(X)==z
            self.thuktong += [thukmai.sum()]
            self.entropy += [self.ha_entropy(X,z)]

        self.w[1:] /= X_std
        self.w[0] -= (self.w[1:]*X_mean).sum()

    def thamnai(self,X):
        return np.dot(X,self.w[1:])+self.w[0]>0

    def ha_sigmoid(self,X):
        return sigmoid(np.dot(X,self.w[1:])+self.w[0])

    def ha_entropy(self,X,z):
        phi = self.ha_sigmoid(X)
        return -(z*np.log(phi+1e-7)+(1-z)*np.log(1-phi+1e-7)).mean()



X,z = datasets.make_blobs(n_samples=10000,n_features=2,centers=2,cluster_std=2,random_state=2)
eta = 0.1
n_thamsam = 100
n_batch = 150
tl = ThotthoiLogistic(eta)
tl.rianru(X,z,n_thamsam,n_batch)

plt.subplot(211)
plt.title(u'เอนโทรปี',fontname='Tahoma')
plt.plot(tl.entropy)
plt.tick_params(labelbottom='off')
plt.subplot(212)
plt.title(u'จำนวนที่ถูก',fontname='Tahoma')
plt.plot(tl.thuktong)

plt.figure(figsize=[6,6])
x_sen = np.array([X[:,0].min(),X[:,0].max()])
y_sen = -(tl.w[0]+tl.w[1]*x_sen)/tl.w[2]
thukmai = tl.thamnai(X)==z
plt.gca(aspect=1,xlim=[X[:,0].min(),X[:,0].max()],ylim=[X[:,1].min(),X[:,1].max()])
plt.plot(x_sen,y_sen,lw=3,zorder=0)
plt.scatter(X[thukmai,0],X[thukmai,1],c=z[thukmai],s=10,edgecolor='k',lw=0.5,cmap='winter')
plt.scatter(X[~thukmai,0],X[~thukmai,1],c=z[~thukmai],s=10,edgecolor='r',cmap='winter')
plt.show()