# -*- coding: utf-8 -*-
'''
โค้ดจากบทความในบล็อก หน้า
https://phyblas.hinaboshi.com/20161219
'''

import numpy as np
import matplotlib.pyplot as plt

class ThotthoiPhahunam1D:
    def __init__(self,eta,deg=2):
        self.eta = eta
        self.deg = deg

    def rianru(self,x,z,d_yut=1e-7,n_thamsam=100000):
        X = np.stack([x**i for i in range(self.deg+1)],1) # ทำให้เป็นพหุนาม
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

    def thamnai(self,x):
        X = np.stack([x**i for i in range(self.deg+1)],1)
        return np.dot(X,self.w)

    def ha_sse(self,h,z):
        return ((h-z)**2).sum()

x = np.linspace(0,1,101) # ตัวแปรต้น
z = 1+3*x-7*x**2+5*x**3+0.1*np.random.randn(101) # ตัวแปรตาม

eta = 0.005 # อัตราการเรียนรู้
deg = 3 # ดีกรี
tp = ThotthoiPhahunam1D(eta,deg)
tp.rianru(x,z)

plt.scatter(x,z) # วาดจุดข้อมูล
x = np.linspace(0,1,101)
h = tp.thamnai(x)
plt.plot(x,h,'b') # วาดกราฟที่คำนวณมาได้
plt.show()


