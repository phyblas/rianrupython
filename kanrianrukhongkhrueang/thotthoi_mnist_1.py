# -*- coding: utf-8 -*-
'''
โค้ดจากบทความในบล็อก หน้า
https://phyblas.hinaboshi.com/20170922
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

def softmax(x):
    exp_x = np.exp(x.T-x.max(1))
    return (exp_x/exp_x.sum(0)).T

class ThotthoiLogistic:
    def __init__(self,eta):
        self.eta = eta

    def rianru(self,X,z,n_thamsam,n_batch=0,romaiphoem=0):
        n = len(z)
        if(n_batch==0 or n<n_batch):
            n_batch = n
        self.kiklum = int(z.max()+1)
        z_1h = z[:,None]==range(self.kiklum)
        self.w = np.zeros([X.shape[1]+1,self.kiklum])
        self.entropy = []
        self.thuktong = []
        thukmaksut = 0 # ค่าจำนวนที่ถูกมากสุด
        thukmaiphoem = 0 # นับว่าจำนวนที่ถูกไม่เพิ่มมาแล้วกี่ครั้ง
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
            thukmak = thukmai.mean()*100
            
            if(thukmak > thukmaksut):
                # ถ้าจำนวนที่ถูกมากขึ้นกว่าเดิมก็บันทึกค่าจำนวนนั้น และน้ำหนักในตอนนั้นไว้
                thukmaksut = thukmak
                thukmaiphoem = 0
                w = self.w.copy()
            else:
                thukmaiphoem += 1 # ถ้าไม่ถูกมากขึ้นก็นับไว้ว่าไม่เพิ่มไปอีกครั้งแล้ว
            
            self.thuktong += [thukmak]
            self.entropy += [self.ha_entropy(X,z_1h)]
            print(u'ครั้งที่ %d ถูก %.3f%% สูงสุด %.3f%% ไม่เพิ่มมาแล้ว %d ครั้ง'%(j+1,self.thuktong[-1],thukmaksut,thukmaiphoem))
            
            if(romaiphoem!=0 and thukmaiphoem>=romaiphoem):
                break # ถ้าจำนวนที่ถูกไม่เพิ่มเลย 10 ครั้งก็เลิกทำ
                
        self.w = w # ค่าน้ำหนักที่ได้ในท้ายสุด เอาตามค่าที่ทำให้ทายถูกมากที่สุด

    def thamnai(self,X):
        return (np.dot(X,self.w[1:])+self.w[0]).argmax(1)

    def ha_softmax(self,X):
        return softmax(np.dot(X,self.w[1:])+self.w[0])

    def ha_entropy(self,X,z_1h):
        return -(z_1h*np.log(self.ha_softmax(X)+1e-7)).mean()

np.random.seed(0)
mnist = datasets.fetch_mldata('MNIST original')
mnist.data = mnist.data/255.
sumriang = np.random.permutation(len(mnist.target))
X = mnist.data[sumriang]
z = mnist.target[sumriang]

# เริ่มการเรียนรู้
eta = 0.24
n_thamsam = 1000
n_batch = 100
romaiphoem = 10
tl = ThotthoiLogistic(eta)
tl.rianru(X,z,n_thamsam,n_batch,romaiphoem)

# กราฟแสดงความคืบหน้าในการเรียนรู้
ax = plt.subplot(211)
ax.set_title(u'เอนโทรปี',fontname='Tahoma')
plt.plot(tl.entropy)
plt.tick_params(labelbottom='off')
ax = plt.subplot(212)
ax.set_title(u'% ถูก',fontname='Tahoma')
plt.plot(tl.thuktong)

# ภาพแสดงน้ำหนักของเลข 0
plt.figure()
plt.imshow(tl.w[1:,0].reshape(28,28),cmap='gray_r')

# ภาพแสดงน้ำหนักของเลข 1~9
plt.figure()
for i in range(1,10):
    plt.subplot(330+i)
    plt.imshow(tl.w[1:,i].reshape(28,28),cmap='gray_r')
plt.show()
