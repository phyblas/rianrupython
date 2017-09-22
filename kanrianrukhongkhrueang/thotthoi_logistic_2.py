# -*- coding: utf-8 -*-
'''
โค้ดจากบทความในบล็อก หน้า
https://phyblas.hinaboshi.com/20161124
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler as Sta

def sigmoid(x):
    return 1/(1+np.exp(-x))

class ThotthoiLogistic:
    def __init__(self,eta):
        self.eta = eta
    # เรียนรู้
    def rianru(self,X,z,n_thamsam):
        self.sse = []
        self.thuktong = []
        self.w = np.zeros(X.shape[1]+1)
        self.sta = Sta()
        # ทำให้เป็นมาตรฐาน
        X = self.sta.fit_transform(X)

        phi = self.ha_sigmoid(X)
        for i in range(n_thamsam):
            # ปรับค่าน้ำหนัก
            eee = 2*phi*(1-phi)*(z-phi)*self.eta
            self.w[1:] += np.dot(X.T,eee)
            self.w[0] += eee.sum()
            phi = self.ha_sigmoid(X)
            # บันทึกผลในแต่ละรอบ
            thukmai = np.abs(phi-z)<0.5
            self.thuktong += [thukmai.sum()]
            self.sse += [self.ha_sse(X,z)]
        # ปรับค่าน้ำหนักให้เข้ากับข้อมูลเดิม
        self.w[1:] /= self.sta.scale_
        self.w[0] -= (self.w[1:]*self.sta.mean_).sum()
    # ทำนายผล
    def thamnai(self,X):
        return self.ha_sigmoid(X)>0.5
    # ฟังก์ชันกระตุ้น
    def ha_sigmoid(self,X):
        return sigmoid(np.dot(X,self.w[1:])+self.w[0])
    # หาค่าเสียหาย
    def ha_sse(self,X,z):
        return ((z-self.ha_sigmoid(X))**2).sum()



eta = 0.00002
n_thamsam = 80000

n_pluk = 600
x_phakkat = np.random.uniform(0,1000,n_pluk)
y_phakkat = np.random.uniform(0,200,n_pluk)
tomai = (3*x_phakkat+y_phakkat-2000>0).astype(int)
xy_phakkat = np.stack([x_phakkat,y_phakkat],axis=1)

tl = ThotthoiLogistic(eta)
tl.rianru(xy_phakkat,tomai,n_thamsam)

print('ได้สมการเส้นแบ่งเขตเป็น %.5fx%+.5fy%+.5f = 0'%(tl.w[1],tl.w[2],tl.w[0]))
print('ทายถูกทั้งหมด %d จาก %d'%(tl.thuktong[-1],len(tomai)))

x_sen = np.array([x_phakkat.min(),x_phakkat.max()])
y_sen = -(tl.w[0]+tl.w[1]*x_sen)/tl.w[2]
thukmai = tl.thamnai(xy_phakkat)==tomai

plt.figure(figsize=[11,3])
plt.gca(aspect=1,xlim=[x_phakkat.min(),x_phakkat.max()],ylim=[y_phakkat.min(),y_phakkat.max()],xlabel='x',ylabel='y')
if(tl.w[1]*tl.w[2]<0):
    plt.fill_between(x_sen,y_sen,[y_phakkat.min(),y_phakkat.min()],color='#33ee33')
else:
    plt.fill_between(x_sen,y_sen,[y_phakkat.max(),y_phakkat.max()],color='#33ee33')
plt.scatter(x_phakkat[thukmai==1],y_phakkat[thukmai==1],c=tomai[thukmai==1],s=50,edgecolor='k',cmap='summer_r')
plt.scatter(x_phakkat[thukmai==0],y_phakkat[thukmai==0],c=tomai[thukmai==0],s=50,edgecolor='r',lw=2,cmap='summer_r')
plt.show()