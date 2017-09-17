# -*- coding: utf-8 -*-

'''https://phyblas.hinaboshi.com/20161207'''

import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    exp_x = np.exp(x.T-x.max(1))
    return (exp_x/exp_x.sum(0)).T

class ThotthoiSoftmax:
    def __init__(self,eta):
        self.eta = eta

    def rianru(self,X,z,n_thamsam):
        self.kiklum = int(z.max()+1)
        X_std = X.std(0)
        X_mean = X.mean(0)
        X = (X-X_mean)/X_std
        z_1h = z[:,None]==range(self.kiklum)
        self.w = np.zeros([X.shape[1]+1,self.kiklum])
        self.entropy = []
        self.thuktong = []
        phi = self.ha_softmax(X)
        for i in range(n_thamsam):
            eee = (z_1h-phi)*self.eta
            self.w[1:] += np.dot(eee.T,X).T
            self.w[0] += eee.sum(0)
            phi = self.ha_softmax(X)
            thukmai = phi.argmax(1)==z
            self.thuktong += [thukmai.sum()]
            self.entropy += [self.ha_entropy(X,z_1h)]
        self.w[1:] /= X_std[:,None]
        self.w[0] -= (self.w[1:]*X_mean[:,None]).sum(0)

    def thamnai(self,X):
        return (np.dot(X,self.w[1:])+self.w[0]).argmax(1)

    def ha_softmax(self,X):
        return softmax(np.dot(X,self.w[1:])+self.w[0])

    def ha_entropy(self,X,z_1h):
        return -(z_1h*np.log(self.ha_softmax(X)+1e-7)).sum()



nueasat = np.random.randint(0,8000,200)
phonlamai = np.random.randint(0,8000,200)
plianrang = np.tile([4],200)
plianrang[nueasat>5000] = 3
plianrang[nueasat-phonlamai*2>-3000] = 2
plianrang[phonlamai<1000] = 1
plianrang[nueasat+phonlamai<4000] = 0

eta = 0.001
n_thamsam = 10000
ahan = np.stack([nueasat,phonlamai],axis=1)
ts = ThotthoiSoftmax(eta)
ts.rianru(ahan,plianrang,n_thamsam)

ax = plt.subplot(211)
ax.set_title(u'เอนโทรปี',fontname='Tahoma')
plt.plot(ts.entropy)
plt.tick_params(labelbottom='off')
ax = plt.subplot(212)
ax.set_title(u'จำนวนที่ถูก',fontname='Tahoma')
plt.plot(ts.thuktong)

plt.figure(figsize=[6,6])
ax = plt.gca(xlim=[0,8000],ylim=[0,8000],aspect=1)
ax.set_xlabel(u'เนื้อสัตว์',fontname='Tahoma')
ax.set_ylabel(u'ผลไม้',fontname='Tahoma')

nmesh = 200
mx,my = np.meshgrid(np.linspace(0,8000,nmesh),np.linspace(0,8000,nmesh))
mx = mx.ravel()
my = my.ravel()
mX = np.stack([mx,my],1)
mz = ts.thamnai(mX)
si = ['#770077','#777700','#007777','#007700','#000077']
c = [si[i] for i in mz]
ax.scatter(mx,my,c=c,s=1,marker='s',alpha=0.3,lw=0)
ax.contour(mx.reshape(nmesh,nmesh),my.reshape(nmesh,nmesh),mz.reshape(nmesh,nmesh),
           ts.kiklum,colors='k',linewidths=3,zorder=0)
thukmai = ts.thamnai(ahan)==plianrang
c = np.array([si[i] for i in plianrang])
ax.scatter(nueasat[thukmai],phonlamai[thukmai],c=c[thukmai],s=100,edgecolor='k')
ax.scatter(nueasat[~thukmai],phonlamai[~thukmai],c=c[~thukmai],s=100,edgecolor='r',lw=2)
plt.show()