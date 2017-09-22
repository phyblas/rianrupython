# -*- coding: utf-8 -*-
'''
โค้ดจากบทความในบล็อก หน้า
https://phyblas.hinaboshi.com/20161210
'''

import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
x = np.arange(1,31,2) # สร้างข้อมูลตัวแปรต้น
z = 2.5+x*0.5+np.random.randn(15)*0.5 # สร้างข้อมูลตัวแปรตาม

eta = 0.0002 # อัตราการเรียนรู้
n_thamsam = 100000 # จำนวนครั้งที่ทำซ้ำสูงสุด
d_yut = 1e-7 # ค่าความเปลี่ยนแปลงน้ำหนักและไบแอสสูงสุดที่จะให้หยุดได้
w,b = 0,0 # น้ำหนักและไบแอส เริ่มต้นเป็น 0
wi,bi = [w],[b] # ลิสต์บันทึกค่าน้ำหนักและไบแอส
h = w*x+b
for i in range(n_thamsam):
    dw = 2*((z-h)*x).sum()*eta
    db = 2*(z-h).sum()*eta
    w += dw
    b += db
    wi += [w] #  บันทึกค่าน้ำหนักและไบแอสใหม่
    bi += [b]
    h = w*x+b
    if(np.abs(dw)<d_yut and np.abs(db)<d_yut):
        break # หยุดเมื่อทั้ง dw และ db ต่ำกว่า d_yut

print('ทำซ้ำไป %d ครั้ง dw=%.3e, db=%.3e'%(i,dw,db))

bi = np.array(bi)
wi = np.array(wi)

# แสดงผลเป็นภาพหุบเขาในสามมิติ
plt.figure(figsize=[8,8])
ax = plt.axes([0,0,1,1],projection='3d',xlabel='b',ylabel='w',zlabel='SSE')
ssei = ((x*wi[:,None]+bi[:,None]-z)**2).sum(1)
ax.plot(bi,wi,ssei,'bo-')
mb,mw = np.meshgrid(np.linspace(0,3,201),np.linspace(0,1.2,201))
sse = ((x*mw.ravel()[:,None]+mb.ravel()[:,None]-z)**2).sum(1).reshape(201,-1)
ax.plot_surface(mb,mw,sse,rstride=5,cstride=5,alpha=0.2,color='b',edgecolor='k')

# แสดงผลเป็นภาพสีในสองมิติ
plt.figure(figsize=[10,5])
plt.pcolormesh(mb,mw,sse,norm=mpl.colors.LogNorm(),cmap='gnuplot')
plt.colorbar(pad=0.01)
plt.plot(bi,wi,'bo-',lw=1)
plt.show()