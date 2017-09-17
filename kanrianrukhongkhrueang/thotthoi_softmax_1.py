# -*- coding: utf-8 -*-

'''https://phyblas.hinaboshi.com/20161205'''

import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    exp_x = np.exp(x.T-x.max(1))
    return (exp_x/exp_x.sum(0)).T

class ThotthoiSoftmax:
    def __init__(self,eta):
        self.eta = eta
    # เรียนรู้
    def rianru(self,X,z,n_thamsam):
        self.kiklum = int(z.max()+1)
        X_std = X.std(0)
        X_mean = X.mean(0)
        # ปรับ X ให้เป็นมาตรฐาน
        X = (X-X_mean)/X_std
        # ทำข้อมูล z ในแบบ one-hot
        z_1h = z[:,None]==range(self.kiklum)
        # ใส่ค่าน้ำหนักเริ่มต้นเป็น 0 ทุกตัว
        self.w = np.zeros([X.shape[1]+1,self.kiklum])
        self.khasiahai = []
        self.thuktong = []
        phi = self.ha_softmax(X)
        #เริ่มการเรียนรู้
        for i in range(n_thamsam):
            # คำนวณและปรับค่าน้ำหนักของแต่ละกลุ่ม
            for n in range(self.kiklum):
                delta_nm = np.zeros(self.kiklum)
                delta_nm[n] = 1
                eee = 2*(phi*(delta_nm-phi[:,n:n+1])*(z_1h-phi)).sum(1)*self.eta
                self.w[1:,n] += (eee[:,None]*X).sum(0)
                self.w[0,n] += eee.sum()
            phi = self.ha_softmax(X)
            # เปรียบเทียบค่าที่ทายได้กับคำตอบจริง
            thukmai = phi.argmax(1)==z
            # บันทึกจำนวนที่ถูกและค่าเสียหาย
            self.thuktong += [thukmai.sum()]
            self.khasiahai += [self.ha_khasiahai(X,z_1h)]
        # ปรับค่าน้ำหนักให้เข้ากับข้อมูลเดิมที่ไม่ได้ปรับมาตรฐาน
        self.w[1:] /= X_std[:,None]
        self.w[0] -= (self.w[1:]*X_mean[:,None]).sum(0)
    # ทำนายว่าอยู่กลุ่มไหน
    def thamnai(self,X):
        return self.ha_softmax(X).argmax(1)
    # หาความน่าจะเป็นที่จะอยู่ในแต่ละกลุ่ม
    def ha_softmax(self,X):
        return softmax(np.dot(X,self.w[1:])+self.w[0])
    # หาค่าเสียหาย
    def ha_khasiahai(self,X,z_1h):
        return ((z_1h-self.ha_softmax(X))**2).sum()

# สุ่มจำนวนผักและปลา
phak = np.random.randint(0,4000,100)
pla = np.random.randint(0,4000,100)
# กำหนดว่าจะเปลี่ยนเป็นร่างไหน
plianrang = np.tile([1],100)
plianrang[phak-pla*2<1000] = 2
plianrang[phak-pla<0] = 3
plianrang[phak*2-pla<-1000] = 4
plianrang[phak+pla<2000] = 0

ahan = np.stack([phak,pla],1)

eta = 0.001 # อัตราการเรียนรู้
n_thamsam = 1000 # จำนวนครั้งที่ทำซ้ำ
ts = ThotthoiSoftmax(eta) # สร้างออบเจ็กต์จากคลาส
ts.rianru(ahan,plianrang,n_thamsam) # เริ่มการเรียนรู้

# วาดกราฟแสดงการเปลี่ยนแปลงของค่าเสียหายและจำนวนที่ถูก
ax = plt.subplot(211)
ax.set_title(u'ผลรวมค่าเสียหาย',fontname='Tahoma')
plt.plot(ts.khasiahai)
ax = plt.subplot(212)
ax.set_title(u'จำนวนที่ถูก',fontname='Tahoma')
plt.plot(ts.thuktong)

# วาดภาพแสดงแผลที่ได้
plt.figure(figsize=[6,6])
ax = plt.gca(xlim=[0,4000],ylim=[0,4000],aspect=1)
ax.set_xlabel(u'ผัก',fontname='Tahoma')
ax.set_ylabel(u'ปลา',fontname='Tahoma')

nmesh = 200
mx,my = np.meshgrid(np.linspace(0,4000,nmesh),np.linspace(0,4000,nmesh))
mx = mx.ravel()
my = my.ravel()
mX = np.stack([mx,my],1)
mz = ts.thamnai(mX)
si = ['#770077','#777700','#007777','#007700','#000077']
for i in range(ts.kiklum):
    ax.scatter(mx[mz==i],my[mz==i],c=si[i],s=2,marker='s',alpha=0.2,lw=0)
ax.contour(mx.reshape(nmesh,nmesh),my.reshape(nmesh,nmesh),mz.reshape(nmesh,nmesh),
           ts.kiklum,colors='k',linewidths=3,zorder=0)
thukmai = ts.thamnai(ahan)==plianrang
c = np.array([si[i] for i in plianrang])
ax.scatter(phak[thukmai],pla[thukmai],c=c[thukmai],s=100,edgecolor='k')
ax.scatter(phak[~thukmai],pla[~thukmai],c=c[~thukmai],s=100,edgecolor='r',lw=2)



# ทดสอบกับข้อมูลกลุ่มก้อนที่สร้างจาก sklearn.datasets.make_blobs
from sklearn import datasets
X,z = datasets.make_blobs(n_samples=1000,n_features=2,centers=6,random_state=36)
eta = 0.001
n_thamsam = 1000
ts = ThotthoiSoftmax(eta)
ts.rianru(X,z,n_thamsam)

plt.figure()
ax = plt.gca(xlim=[X[:,0].min(),X[:,0].max()],ylim=[X[:,1].min(),X[:,1].max()],aspect=1)
nmesh = 200
mx,my = np.meshgrid(np.linspace(X[:,0].min(),X[:,0].max(),nmesh),np.linspace(X[:,1].min(),X[:,1].max(),nmesh))
mx = mx.ravel()
my = my.ravel()
mX = np.stack([mx,my],1)
mz = ts.thamnai(mX)
si = ['#770077','#777700','#007777','#007700','#000077','#770000']
for i in range(ts.kiklum):
    ax.scatter(mx[mz==i],my[mz==i],c=si[i],s=2,marker='s',alpha=0.2,lw=0)
ax.contour(mx.reshape(nmesh,nmesh),my.reshape(nmesh,nmesh),mz.reshape(nmesh,nmesh),
           ts.kiklum,colors='k',linewidths=3,zorder=0)
thukmai = ts.thamnai(X)==z
c = np.array([si[i] for i in z])
ax.scatter(X[thukmai,0],X[thukmai,1],c=c[thukmai],s=100,edgecolor='k')
ax.scatter(X[~thukmai,0],X[~thukmai,1],c=c[~thukmai],s=100,edgecolor='r',lw=2)
plt.show()