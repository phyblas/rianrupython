# -*- coding: utf-8 -*-

'''https://phyblas.hinaboshi.com/20161212'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class ThotthoiChoengsen:
    def __init__(self,eta):
        self.eta = eta
    # เรียนรู้จากข้อมูล X และ z ที่ใส่เข้าไป
    def rianru(self,X,z,d_yut=1e-7,n_thamsam=100000):
        self.w = np.zeros(X.shape[1]+1)
        dw = np.zeros(X.shape[1]+1)
        h = self.thamnai(X)
        self.sse = [self.ha_sse(h,z)] # ลิสต์บันทึกค่า SSE ในแต่ละรอบ
        for i in range(n_thamsam):
            eee = 2*(z-h)*self.eta
            dw[1:] = np.dot(eee,X)
            dw[0] = eee.sum()
            self.w += dw
            h = self.thamnai(X)
            self.sse += [self.ha_sse(h,z)]
            if(np.all(abs(dw)<d_yut)):
                break
    # ทำนายค่าจาก X ที่ใส่เข้าไป
    def thamnai(self,X):
        return np.dot(X,self.w[1:])+self.w[0]
    # หาค่าผลรวมความคลาดเคลื่อนกำลังสอง
    def ha_sse(self,h,z):
        return ((h-z)**2).sum()

phonlamai = np.random.uniform(0,10,100) # ปริมาณผลไม้
phak = np.random.uniform(0,10,100) # ปริมาณผัก
chomti = 10+phonlamai*2+phak*3+np.random.randn(100)*3 # คำนวณพลังโจมตี

# วาดกราฟแท่งแสดงค่า
plt.figure(figsize=[8,8])
ax = plt.axes([0,0,1,1],projection='3d')
ax.set_xlabel(u'ผลไม้',fontname='Tahoma',fontsize=20)
ax.set_ylabel(u'ผัก',fontname='Tahoma',fontsize=20)
ax.set_zlabel(u'พลังโจมตี',fontname='Tahoma',fontsize=20)

# กำหนดสีของเส้นและจุดตามค่า
def si(x):
    x = (x-chomti.min())/(chomti.max()-chomti.min())
    return plt.get_cmap('jet')(x)

# วาดเส้นลากจากพื้นให้กับทุกจุด
for i in range(100):
    ax.plot([phonlamai[i],phonlamai[i]],[phak[i],phak[i]],[0,chomti[i]],color=si(chomti[i]))

ax.scatter(phonlamai,phak,chomti,c=si(chomti),edgecolor='k') # วาดจุด



eta = 0.0001 # อัตราการเรียนรู้
ahan = np.stack([phonlamai,phak],1)
tc = ThotthoiChoengsen(eta) # สร้างออบเจ็กต์
tc.rianru(ahan,chomti) # เริ่มการเรียนรู้

# วาดกราฟแสดงระนาบผลลัพทธ์
plt.figure(figsize=[8,8])
ax = plt.axes([0,0,1,1],projection='3d')
ax.set_xlabel(u'ผลไม้',fontname='Tahoma',fontsize=20)
ax.set_ylabel(u'ผัก',fontname='Tahoma',fontsize=20)
ax.set_zlabel(u'พลังโจมตี',fontname='Tahoma',fontsize=20)
mx,my = np.meshgrid(np.linspace(0,10,11),np.linspace(0,10,11))
mX = np.stack([mx.ravel(),my.ravel()],1)
mz = tc.thamnai(mX).reshape(11,11)
ax.plot_surface(mx,my,mz,rstride=1,cstride=1,alpha=0.2,color='b',edgecolor='k')
h = tc.thamnai(ahan)
for i in range(100):
    ax.plot([phonlamai[i],phonlamai[i]],[phak[i],phak[i]],[h[i],chomti[i]],'k')
ax.scatter(phonlamai,phak,chomti,c=chomti,edgecolor='k',cmap='jet')
print(tc.w) # ค่าน้ำหนักและไบแอสที่ได้

# กราฟแสงความคืบหน้าในการเรียนรู้
plt.figure()
plt.gca(yscale='log',xlim=[-len(tc.sse)*0.01,len(tc.sse)*1.01])
plt.plot(tc.sse)

plt.show()