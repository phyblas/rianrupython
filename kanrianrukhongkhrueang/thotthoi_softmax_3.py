# -*- coding: utf-8 -*-

'''https://phyblas.hinaboshi.com/20161228'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

def softmax(x):
    exp_x = np.exp(x.T-x.max(1))
    return (exp_x/exp_x.sum(0)).T

class ThotthoiSoftmax:
    def __init__(self,eta):
        self.eta = eta

    def rianru(self,X,z,n_thamsam,n_batch=0):
        n = len(z)
        if(n_batch==0 or n<n_batch):
            n_batch = n
        self.kiklum = int(z.max()+1)
        X_std = X.std(0)
        X_mean = X.mean(0)
        X = (X-X_mean)/X_std
        z_1h = z[:,None]==range(self.kiklum)
        self.w = np.zeros([X.shape[1]+1,self.kiklum])
        self.entropy = []
        self.thuktong = []
        for j in range(n_thamsam):
            lueak = np.random.permutation(n)
            for i in range(0,n,n_batch):
                Xn = X[lueak[i:i+n_batch]]
                zn = z_1h[lueak[i:i+n_batch]]
                phi = self.ha_softmax(Xn)
                eee = (zn-phi)/len(zn)*self.eta
                self.w[1:] += np.dot(eee.T,Xn).T
                self.w[0] += eee.sum(0)
            thukmai = self.thamnai(X)==z
            self.thuktong += [thukmai.sum()]
            self.entropy += [self.ha_entropy(X,z_1h)]

        self.w[1:] /= X_std[:,None]
        self.w[0] -= (self.w[1:]*X_mean[:,None]).sum(0)

    def thamnai(self,X):
        return (np.dot(X,self.w[1:])+self.w[0]).argmax(1)

    def ha_softmax(self,X):
        return softmax(np.dot(X,self.w[1:])+self.w[0])

    def ha_entropy(self,X,z_1h):
        return -(z_1h*np.log(self.ha_softmax(X)+1e-7)).mean()



X,z = datasets.make_blobs(n_samples=10000,n_features=2,centers=5,cluster_std=0.8,random_state=1)
eta = 0.1
n_thamsam = 100
n_batch = 150
ts = ThotthoiSoftmax(eta)
ts.rianru(X,z,n_thamsam,n_batch)

ax = plt.subplot(211)
ax.set_title(u'เอนโทรปี',fontname='Tahoma')
plt.plot(ts.entropy)
plt.tick_params(labelbottom='off')
ax = plt.subplot(212)
ax.set_title(u'จำนวนที่ถูก',fontname='Tahoma')
plt.plot(ts.thuktong)

plt.figure(figsize=[6,6])
ax = plt.gca(xlim=[X[:,0].min(),X[:,0].max()],ylim=[X[:,1].min(),X[:,1].max()],aspect=1)

nmesh = 200
mx,my = np.meshgrid(np.linspace(X[:,0].min(),X[:,0].max(),nmesh),np.linspace(X[:,1].min(),X[:,1].max(),nmesh))
mX = np.stack([mx.ravel(),my.ravel()],1)
mz = ts.thamnai(mX)
si = ['#770077','#777700','#007777','#007700','#000077']
c = [si[i] for i in mz]
ax.scatter(mX[:,0],mX[:,1],c=c,s=1,marker='s',alpha=0.3,lw=0)
ax.contour(mx,my,mz.reshape(nmesh,nmesh),
           ts.kiklum,colors='k',lw=2,zorder=0)
thukmai = ts.thamnai(X)==z
c = np.array([si[i] for i in z])
ax.scatter(X[thukmai,0],X[thukmai,1],c=c[thukmai],s=10,edgecolor='k',lw=0.5)
ax.scatter(X[~thukmai,0],X[~thukmai,1],c=c[~thukmai],s=10,edgecolor='r')
plt.show()