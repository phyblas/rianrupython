# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

def softmax(x):
    exp_x = np.exp(x.T-x.max(1))
    return (exp_x/exp_x.sum(0)).T

class ThotthoiSoftmax:
    def __init__(self,eta):
        self.eta = eta

    def rianru(self,X,z,n_thamsam,n_batch=0,X_thotsop=0,z_thotsop=0,romaiphoem=0):
        n = len(z)
        if(type(X_thotsop)!=np.ndarray):
            X_thotsop,z_thotsop = X,z
        if(n_batch==0 or n<n_batch):
            n_batch = n
        self.kiklum = int(z.max()+1)
        z_1h = z[:,None]==range(self.kiklum)
        self.w = np.zeros([X.shape[1]+1,self.kiklum])
        self.entropy = []
        self.thuktong = []
        self.thuktong_thotsop = []
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
            thukmai = self.thamnai(X_thotsop)==z_thotsop
            thukmak_thotsop = thukmai.mean()*100
            
            if(thukmak_thotsop > thukmaksut):
                # ถ้าจำนวนที่ถูกมากขึ้นกว่าเดิมก็บันทึกค่าจำนวนนั้น และน้ำหนักในตอนนั้นไว้
                thukmaksut = thukmak_thotsop
                thukmaiphoem = 0
                w = self.w.copy()
            else:
                thukmaiphoem += 1 # ถ้าไม่ถูกมากขึ้นก็นับไว้ว่าไม่เพิ่มไปอีกครั้งแล้ว
            
            self.thuktong += [thukmak]
            self.thuktong_thotsop += [thukmak_thotsop]
            self.entropy += [self.ha_entropy(X,z_1h)]
            print(u'ครั้งที่ %d ถูก %.3f%% สูงสุด %.3f%% ไม่เพิ่มมาแล้ว %d ครั้ง'%(j+1,self.thuktong_thotsop[-1],thukmaksut,thukmaiphoem))
            
            if(romaiphoem!=0 and thukmaiphoem>=romaiphoem):
                break # ถ้าจำนวนที่ถูกไม่เพิ่มเลย 10 ครั้งก็เลิกทำ
                
        self.w = w # ค่าน้ำหนักที่ได้ในท้ายสุด เอาตามค่าที่ทำให้ทายถูกมากที่สุด

    def thamnai(self,X):
        return (np.dot(X,self.w[1:])+self.w[0]).argmax(1)

    def ha_softmax(self,X):
        return softmax(np.dot(X,self.w[1:])+self.w[0])

    def ha_entropy(self,X,z_1h):
        return -(z_1h*np.log(self.ha_softmax(X)+1e-7)).mean()

# ดึงข้อมูล MNIST
mnist = datasets.fetch_mldata('MNIST original')
X,z = mnist.data,mnist.target
X = X/255.
np.random.seed(0)
X_fuek,X_thotsop,z_fuek,z_thotsop = train_test_split(X,z,test_size=0.2)

# เริ่มการเรียนรู้
eta = 0.24 # อัตราการเรียนรู้
n_thamsam = 1000 # จำนวนทำซ้ำสูงสุดถ้าไม่มีการหยุดเสียก่อน
n_batch = 100 # จำนวนมินิแบตช์
romaiphoem = 10 # จะให้หยุดเมื่อความแม่นยำไม่เพิ่มเกินกี่ครั้ง
ts = ThotthoiSoftmax(eta)
ts.rianru(X_fuek,z_fuek,n_thamsam,n_batch,X_thotsop,z_thotsop,romaiphoem)

# กราฟแสดงความคืบหน้าในการเรียนรู้
ax = plt.subplot(211)
ax.set_title(u'เอนโทรปี',fontname='Tahoma')
plt.plot(ts.entropy)
plt.tick_params(labelbottom='off')
ax = plt.subplot(212)
ax.set_title(u'% ถูก',fontname='Tahoma')
plt.plot(ts.thuktong)
plt.plot(ts.thuktong_thotsop)
plt.show()