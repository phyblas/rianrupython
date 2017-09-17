# -*- coding: utf-8 -*-

'''https://phyblas.hinaboshi.com/20161103'''

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

class ThotthoiLogistic:
    def __init__(self,eta):
        self.eta = eta

    def rianru(self,X,z,n_thamsam):
        self.sse = []
        self.thuktong = []
        self.w = np.zeros(X.shape[1]+1)

        phi = self.ha_sigmoid(X)
        for i in range(n_thamsam):
            eee = 2*phi*(1-phi)*(z-phi)
            self.w[1:] += np.dot(X.T,eee)*self.eta
            self.w[0] += eee.sum()*self.eta
            phi = self.ha_sigmoid(X)
            thukmai = np.abs(phi-z)<0.5
            self.thuktong += [thukmai.sum()]
            self.sse += [self.ha_sse(X,z)]

    def thamnai(self,X):
        return self.ha_sigmoid(X)>0.5

    def ha_sigmoid(self,X):
        return sigmoid(np.dot(X,self.w[1:])+self.w[0])

    def ha_sse(self,X,z):
        return ((z-self.ha_sigmoid(X))**2).sum()

eta = 0.0001 # อัตราการเรียนรู้
n_thamsam = 10000 # จำนวนครั้งที่ทำซ้ำ
x_thua = np.random.uniform(0,40,1000) # สุ่มค่า x ตั้งแต่ 0 ถึง 40 มา 1000 ค่า
y_thua = np.random.uniform(0,30,1000) # สุ่มค่า y ตั้งแต่ 0 ถึง 30 มา 1000 ค่า
ngokmai = np.array(x_thua-2*y_thua+10>0,dtype=int) # กำหนดว่าจะงอกมั้ยโดยดูจากค่าตำแหน่งแกน x และ y

# แสดงการกระจายของข้อมูล
plt.gca(aspect=1,xlim=[0,40],ylim=[0,30],xlabel='x',ylabel='y')
plt.scatter(x_thua,y_thua,c=ngokmai,s=50,edgecolor='k',cmap='summer_r')



# กราฟแสดงค่าของฟังก์ชันซิกมอยด์
x,y = np.meshgrid(np.linspace(0,40,160),np.linspace(0,30,120))
plt.figure(figsize=[8,3])
plt.axes([0.05,0.03,0.4,0.95],aspect=1,title='$sigmoid(x-2y+10)$',xlim=[0,40],ylim=[0,30])
plt.pcolormesh(x,y,sigmoid(x-2*y+10),cmap='coolwarm_r')
plt.axes([0.5,0.03,0.4,0.95],aspect=1,title='$x-2y+10>0$',xlim=[0,40],ylim=[0,30])
plt.pcolormesh(x,y,x-2*y+10>0,cmap='coolwarm_r')
plt.colorbar(cax=plt.axes([0.92,0.03,0.03,0.93]))



# สร้างออบเจ็กต์
tl = ThotthoiLogistic(eta)
# ทำการรวมอาเรย์ของ x และ y ได้เป็นอาเรย์สองมิติ
xy_thua = np.stack([x_thua,y_thua],axis=1)
# เริ่มการเรียนรู้
tl.rianru(xy_thua,ngokmai,n_thamsam)
print('ได้สมการเส้นแบ่งเขตเป็น %.3fx%+.3fy%+.3f = 0'%(tl.w[1],tl.w[2],tl.w[0]))
print('ทายถูกทั้งหมด %d จาก %d'%(tl.thuktong[-1],len(ngokmai)))



# วาดกราฟแสดงค่าความคลาดเคลื่อนและจำนวนที่ถูก
plt.figure()
ax = plt.subplot(211)
ax.set_title(u'ผลรวมความคลาดเคลื่อนกำลังสอง (SSE)',fontname='Tahoma')
plt.plot(tl.sse)
ax = plt.subplot(212)
ax.set_title(u'จำนวนที่ถูก',fontname='Tahoma')
plt.plot(tl.thuktong)



# วาดกราฟแสดงการเปรียบเทียบค่าอัตราการเรียนรู้ที่ต่างกัน
plt.figure()
plt.title(u'จำนวนที่ถูก',fontname='Tahoma')
for eta in [0.01,0.001,0.0001,0.00001]:
    tl = ThotthoiLogistic(eta)
    tl.rianru(xy_thua,ngokmai,10001)
    plt.plot(np.arange(0,10001,100),tl.thuktong[::100],label='%.5f'%eta)
    plt.legend(loc=0)
plt.show()