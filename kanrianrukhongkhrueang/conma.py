# -*- coding: utf-8 -*-
'''
โค้ดจากบทความในบล็อก หน้า
https://phyblas.hinaboshi.com/20170926
'''

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def confusion_matrix(z1,z2):
    '''สร้างเมทริกซ์ความสับสน'''
    n = max(z1.max(),z2.max())+1
    return np.dot((z1==np.arange(n)[:,None]).astype(int),(z2[:,None]==np.arange(n)).astype(int))

def plotconma(conma,log=0):
    '''นำเมทริกซ์ความสับสนมาใส่สี'''
    n = len(conma)
    plt.figure(figsize=[9,8])
    plt.gca(xticks=np.arange(n),xticklabels=np.arange(n),yticks=np.arange(n),yticklabels=np.arange(n))
    plt.xlabel(u'ทายได้',fontname='Tahoma',size=16)
    plt.ylabel(u'คำตอบ',fontname='Tahoma',size=16)
    for i in range(n):
        for j in range(n):
            plt.text(j,i,conma[i,j],ha='center',va='center',size=14)
    if(log):
        plt.imshow(conma,cmap='autumn_r',norm=mpl.colors.LogNorm())
    else:
        plt.imshow(conma,cmap='autumn_r')
    plt.colorbar(pad=0.01)
    plt.show()