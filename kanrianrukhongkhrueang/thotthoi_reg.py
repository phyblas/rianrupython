# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

def softmax(x):
    exp_x = np.exp(x.T-x.max(1))
    return (exp_x/exp_x.sum(0)).T

class ThotthoiLogistic:
    def __init__(self,eta,reg='l2',l=0):
        self.eta = eta
        self.reg = reg
        self.l = l

    def rianru(self,X,z,n_thamsam,n_batch=0,X_truat=0,z_truat=0,romaiphoem=0):
        n = len(z)
        if(type(X_truat)!=np.ndarray): # ถ้าไม่ได้ป้อนข้อมูลตรวจสอบมาด้วย ก็ให้ใช้ข้อมูลฝึกฝนเป็นข้อมูลตรวจสอบ
            X_truat,z_truat = X,z
        if(n_batch==0 or n<n_batch):
            n_batch = n
        self.kiklum = int(z.max()+1) # จำนวนผลลัพธ์
        z_1h = z[:,None]==range(self.kiklum) # แปลงเป็น one-hot
        self.w = np.zeros([X.shape[1]+1,self.kiklum]) # ค่าน้ำหนักตั้งต้นเป็น 0
        self.dw = self.w.copy() # สร้างอาเรย์สำหรับพักค่าการเปลี่ยนแปลงน้ำหนักด้วย
        
        self.khasiahai = [] # ลิสต์บันทึกค่าเสียหาย (เอนโทรปี+เรกูลาไรซ์)
        self.maen_fuek = [] # ลิสต์บันทึกค่าความแม่นในการทำนายข้อมุลฝึก
        self.maen_truat = [] # ลิสต์บันทึกค่าความแม่นในการทำนายข้อมุลตรวจสอบ
        disut = 0 # ค่าจำนวนที่ถูกมากสุด
        maiphoem = 0 # นับว่าจำนวนที่ถูกไม่เพิ่มมาแล้วกี่ครั้ง
        for j in range(n_thamsam):
            lueak = np.random.permutation(n)
            for i in range(0,n,n_batch):
                Xn = X[lueak[i:i+n_batch]]
                zn = z_1h[lueak[i:i+n_batch]]
                phi = self.ha_softmax(Xn)
                eee = (zn-phi)/len(zn)
                self.dw[1:] = np.dot(eee.T,Xn).T
                self.dw[0] = eee.sum(0)
                # หาก l ไม่เป็น 0 ให้ปรับค่าน้ำหนักตามการเรกูลาไรซ์
                if(self.l>0):
                    if(self.reg=='l1'):
                        self.dw[1:] -= (self.w[1:]!=0)*np.where(self.w[1:]>0,1,-1)*self.l/n
                    else: # l2
                        self.dw[1:] -= 2*self.w[1:]*self.l/n
                self.w += self.dw*self.eta
            
            thukmai = self.thamnai(X)==z
            maen_fuek = thukmai.mean()*100
            thukmai = self.thamnai(X_truat)==z_truat
            maen_truat = thukmai.mean()*100
            khasiahai = self.ha_entropy(X,z_1h) # ค่าเสียหาย
            # หาก l ไม่เป็น 0 ให้บวกเรกูลาไรซ์เข้าไปในค่าเสียหายด้วย
            if(l!=0):
                if(reg=='l1'):
                    khasiahai += l*np.abs(self.w[1:]).sum()
                else:
                    khasiahai += l*((self.w[1:])**2).sum()
            
            if(maen_truat > disut):
                # ถ้าจำนวนที่ถูกมากขึ้นกว่าเดิมก็บันทึกค่าจำนวนนั้น และน้ำหนักในตอนนั้นไว้
                disut = maen_truat
                maiphoem = 0
                w = self.w.copy()
            else:
                maiphoem += 1 # ถ้าไม่ถูกมากขึ้นก็นับไว้ว่าไม่เพิ่มไปอีกครั้งแล้ว
            
            self.maen_fuek.append(maen_fuek)
            self.maen_truat.append(maen_truat)
            self.khasiahai.append(khasiahai)
            print(u'ครั้งที่ %d ถูก %.3f%% สูงสุด %.3f%% ไม่เพิ่มมาแล้ว %d ครั้ง'%(j+1,self.maen_truat[-1],disut,maiphoem))
            
            if(romaiphoem!=0 and maiphoem>=romaiphoem):
                break # ถ้าจำนวนที่ถูกไม่เพิ่มเลย 10 ครั้งก็เลิกทำ
                
        self.w = w # ค่าน้ำหนักที่ได้ในท้ายสุด เอาตามค่าที่ทำให้ทายถูกมากที่สุด

    def thamnai(self,X):
        return (np.dot(X,self.w[1:])+self.w[0]).argmax(1)

    def ha_softmax(self,X):
        return softmax(np.dot(X,self.w[1:])+self.w[0])

    def ha_entropy(self,X,z_1h):
        return -(z_1h*np.log(self.ha_softmax(X)+1e-7)).mean()

# ดึงข้อมูล MNIST
np.random.seed(0)
mnist = datasets.fetch_mldata('MNIST original')
X,z = mnist.data,mnist.target
sumriang = np.random.permutation(len(mnist.target))
X = mnist.data[sumriang[:700]]
z = mnist.target[sumriang[:700]]
X = X/255.
X_fuek,X_truat,z_fuek,z_truat = train_test_split(X,z,test_size=0.2)

# เริ่มการเรียนรู้
eta = 0.2 # อัตราการเรียนรู้
n_thamsam = 100 # จำนวนทำซ้ำสูงสุดถ้าไม่มีการหยุดเสียก่อน
n_batch = 100 # จำนวนมินิแบตช์
reg = 'l2'
l = 80
tl = ThotthoiLogistic(eta,reg,l)
tl.rianru(X_fuek,z_fuek,n_thamsam,n_batch,X_truat,z_truat)

# กราฟแสดงความคืบหน้าในการเรียนรู้
plt.plot(tl.maen_fuek,'#dd0000')
plt.plot(tl.maen_truat,'#00aa33')
plt.legend([u'ฝึกฝน',u'ตรวจสอบ'],prop={'family':'Tahoma'})
plt.figure()
w = tl.w[1:]
si = plt.get_cmap('seismic')(w/np.abs(w).max()/2+0.5)
for i in range(1,10):
    plt.subplot(330+i)
    plt.imshow(si[:,i].reshape(28,28,4))
plt.show()
print('%.3f%% / %.3f%%'%(max(tl.maen_fuek),max(tl.maen_truat)))