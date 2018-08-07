# -*- coding: utf-8 -*-
'''
โค้ดจากบทความในบล็อก หน้า
https://phyblas.hinaboshi.com/20161207
'''

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

class ThotthoiLogistic:
    def __init__(self,eta):
        self.eta = eta

    def rianru(self,X,z,n_thamsam):
        X_std = X.std()
        X_mean = X.mean()
        self.entropy = []
        self.thuktong = []
        self.w = np.zeros(X.shape[1]+1)
        X = (X-X_mean)/X_std
        phi = self.ha_sigmoid(X)
        for i in range(n_thamsam):
            eee = (z-phi)*self.eta
            self.w[1:] += np.dot(X.T,eee)
            self.w[0] += eee.sum()
            phi = self.ha_sigmoid(X)
            thukmai = np.abs(phi-z)<0.5
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
        return -(z*np.log(phi+1e-7)+(1-z)*np.log(1-phi+1e-7)).sum()



x_manfarang = np.random.uniform(0,200,1000)
y_manfarang = np.random.uniform(0,160,1000)
yaimai = (2*x_manfarang+y_manfarang-300>0).astype(int)

eta = 0.001
n_thamsam = 10000
xy_manfarang = np.stack([x_manfarang,y_manfarang],axis=1)
tl = ThotthoiLogistic(eta)
tl.rianru(xy_manfarang,yaimai,n_thamsam)

ax = plt.subplot(211)
ax.set_title(u'เอนโทรปี',fontname='Tahoma')
plt.plot(tl.entropy)
plt.tick_params(labelbottom='off')
ax = plt.subplot(212)
ax.set_title(u'จำนวนที่ถูก',fontname='Tahoma')
plt.plot(tl.thuktong)

x_sen = np.array([0,200])
y_sen = -(tl.w[0]+tl.w[1]*x_sen)/tl.w[2]
thukmai = tl.thamnai(xy_manfarang)==yaimai
plt.show()

plt.figure(figsize=[8,6])
plt.gca(aspect=1,xlim=[0,200],ylim=[0,160],xlabel='x',ylabel='y')
if(tl.w[1]*tl.w[2]<0):
    plt.fill_between(x_sen,y_sen,[0,0],color='#66ee99')
else:
    plt.fill_between(x_sen,y_sen,[200,160],color='#66ee99')
plt.scatter(x_manfarang[thukmai],y_manfarang[thukmai],c=yaimai[thukmai],s=50,edgecolor='k',cmap='summer_r')
plt.scatter(x_manfarang[~thukmai],y_manfarang[~thukmai],c=yaimai[~thukmai],s=50,edgecolor='r',lw=2,cmap='summer_r')
plt.show()