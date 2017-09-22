# -*- coding: utf-8 -*-
'''
โค้ดจากบทความในบล็อก หน้า
https://phyblas.hinaboshi.com/20161219
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures as Pf

class ThotthoiPhahunam:
    def __init__(self,eta,deg=2):
        self.eta = eta
        self.deg = deg

    def rianru(self,X,z,d_yut=1e-7,n_thamsam=100000):
        X = Pf(self.deg).fit_transform(X) # ทำให้เป็นพหุนาม
        self.w = np.zeros(X.shape[1])
        dw = np.zeros(X.shape[1])
        h = np.dot(X,self.w)
        self.sse = [self.ha_sse(h,z)]
        for i in range(n_thamsam):
            eee = 2*(z-h)*self.eta
            dw = np.dot(eee,X)
            self.w += dw
            h = np.dot(X,self.w)
            self.sse += [self.ha_sse(h,z)]
            if(np.all(np.abs(dw)<d_yut)):
                break

    def thamnai(self,X):
        X = Pf().fit_transform(X)
        return np.dot(X,self.w)

    def ha_sse(self,h,z):
        return ((h-z)**2).sum()

x = np.random.uniform(0,4,500)
y = np.random.uniform(0,4,500)
z = 22 + 4*x + 7*y + x*x - 6*x*y + y*y + np.random.randn(500)

# วาดกราฟแท่งแสดงการกระจายข้อมูล
plt.figure(figsize=[8,8])
ax = plt.axes([0,0,1,1],projection='3d',xlim=[0,4],ylim=[0,4],zlim=[z.min(),z.max()])
for i in range(len(z)):
    ax.plot([x[i],x[i]],[y[i],y[i]],[z[i],0],c=plt.get_cmap('jet')((z[i]-z.min())/(z.max()-z.min())),alpha=0.2)
ax.scatter(x,y,z,c=z,edgecolor='k')

X = np.stack([x,y],1) # รวมตัวแปรต้นเข้าด้วยกัน
eta = 0.00001
deg = 2
tp = ThotthoiPhahunam(eta,deg)
tp.rianru(X,z)

# วาดพื้นผิวผลลัพธ์ที่ได้
plt.figure(figsize=[8,8])
ax = plt.axes([0,0,1,1],projection='3d',xlim=[0,4],ylim=[0,4],zlim=[z.min(),z.max()])
mx,my = np.meshgrid(np.linspace(0,4,11),np.linspace(0,4,11))
mX = np.stack([mx.ravel(),my.ravel()],1)
mz = tp.thamnai(mX).reshape(11,11)
ax.plot_surface(mx,my,mz,rstride=1,cstride=1,alpha=0.2,color='b',edgecolor='k')
h = tp.thamnai(X)
for i in range(len(z)):
    ax.plot([x[i],x[i]],[y[i],y[i]],[h[i],z[i]],'k')
ax.scatter(x,y,z,c=np.abs(z-h),edgecolor='k',cmap='jet')
plt.show()