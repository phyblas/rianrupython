# coding: utf-8
'''
รวมคลาสและฟังก์ชันที่ใช้ในบทเรียนโครงข่ายประสาทเทียมเบื้องต้น
https://phyblas.hinaboshi.com/saraban/khrong_khai_prasat_thiam
'''
import numpy as np

def ha_1h(z,n,i=0):
    Z = z[:,None]==range(n)
    if(i):
        return Z.astype(int)
    return Z



class Tuaprae(object):
    def __init__(self,kha,thima=None):
        self.kha = kha
        self.thima = thima
    
    def phraeyon(self,g=1):
        if(self.thima is None):
            return
        g = self.thima.yon(g)
        for tpt in self.thima.tuapraeton:
            tpt.phraeyon(g)

class Chan:
    def __init__(self):
        'รอนิยามในคลาสย่อย'
    
    def __call__(self,*tuaprae):
        self.tuapraeton = []
        kha_tuapraeton = []
        for tp in tuaprae:
            if(type(tp)==Tuaprae):
                self.tuapraeton.append(tp)
                kha_tuapraeton.append(tp.kha)
            else:
                kha_tuapraeton.append(tp)
        kha_tuapraetam = self.pai(*kha_tuapraeton)
        tuapraetam = Tuaprae(kha_tuapraetam,thima=self)
        return tuapraetam
    
    def pai(self):
        'รอนิยามในคลาสย่อย'
    def yon(self):
        'รอนิยามในคลาสย่อย'

class Param:
    def __init__(self,kha):
        self.kha = kha
        self.g = 0

class Affin(Chan):
    def __init__(self,m0,m1,sigma=0.1):
        self.m = m0,m1
        self.param = [Param(np.random.normal(0,sigma,self.m)),
                      Param(np.zeros(m1))]
    
    def pai(self,X):
        self.X = X
        return np.dot(X,self.param[0].kha) + self.param[1].kha
    
    def yon(self,g):
        self.param[0].g += np.dot(self.X.T,g)
        self.param[1].g += g.sum(0)
        return np.dot(g,self.param[0].kha.T)

class Sigmoid(Chan):
    def pai(self,a):
        self.h = 1/(1+np.exp(-a))
        return self.h
    
    def yon(self,g):
        return g*(1.-self.h)*self.h

class Relu(Chan):
    def pai(self,x):
        self.krong = (x>0)
        return np.where(self.krong,x,0)
    
    def yon(self,g):
        return np.where(self.krong,g,0)

class Softmax_entropy(Chan):
    def pai(self,a,Z):
        self.Z = Z
        exp_a = np.exp(a.T-a.max(1))
        self.h = (exp_a/exp_a.sum(0)).T
        return -(np.log(self.h[Z]+1e-10)).mean()
    
    def yon(self,g):
        return g*(self.h-self.Z)/len(self.h)

class Sigmoid_entropy(Chan):
    def pai(self,a,z):
        self.z = z[:,None]
        self.h = 1/(1+np.exp(-a))
        return -(self.z*np.log(self.h+1e-17)+(1-self.z)*np.log(1-self.h+1e-17)).mean()
    
    def yon(self,g):
        return g*(self.h-self.z)/len(self.h)

class Mse(Chan):
    def pai(self,h,z):
        self.z = z[:,None]
        self.h = h
        return ((self.z-h)**2).mean()
    
    def yon(self,g):
        return g*2*(self.h-self.z)/len(self.z)



class Sgd:
    def __init__(self,param,eta=0.01):
        self.param = param
        self.eta = eta

    def __call__(self):
        for p in self.param:
            p.kha -= self.eta*p.g
            p.g = 0

class Mmtsgd:
    def __init__(self,param,eta=0.01,mmt=0.9):
        self.param = param
        self.eta = eta
        self.mmt = mmt
        self.d = [0]*len(param)

    def __call__(self):
        for i,p in enumerate(self.param):
            self.d[i] = self.mmt*self.d[i]-self.eta*p.g
            p.kha += self.d[i]
            p.g = 0

class Nag:
    def __init__(self,param,eta=0.01,mmt=0.9):
        self.param = param
        self.eta = eta
        self.mmt = mmt
        self.d = [0]*len(param)
        self.g0 = np.nan

    def __call__(self):
        if(self.g0 is np.nan):
            self.g0 = [p.g for p in self.param]
        for i,p in enumerate(self.param):
            self.d[i] = self.mmt*self.d[i]-self.eta*(p.g+self.mmt*(p.g-self.g0[i]))
            self.g0[i] = p.g
            p.kha += self.d[i]
            p.g = 0

class Adagrad:
    def __init__(self,param,eta=0.01):
        self.param = param
        self.eta = eta
        self.G = [1e-7]*len(param)

    def __call__(self):
        for i,p in enumerate(self.param):
            self.G[i] += p.g**2
            p.kha += -self.eta*p.g/np.sqrt(self.G[i])
            p.g = 0

class Adadelta:
    def __init__(self,param,eta=0.01,rho=0.95):
        self.param = param
        self.eta = eta
        self.rho = rho
        self.G = [1e-7]*len(param)

    def __call__(self):
        for i,p in enumerate(self.param):
            self.G[i] = self.rho*self.G[i]+(1-self.rho)*p.g**2
            p.kha += -self.eta*p.g/np.sqrt(self.G[i])
            p.g = 0

class Adam:
    def __init__(self,param,eta=0.001,beta1=0.9,beta2=0.999):
        self.param = param
        self.eta = eta
        self.beta1 = beta1
        self.beta2 = beta2
        n = len(param)
        self.m = [0]*n
        self.v = [1e-7]*n
        self.t = 1

    def __call__(self):
        for i,p in enumerate(self.param):
            self.m[i] = self.beta1*self.m[i]+(1-self.beta1)*p.g
            self.v[i] = self.beta2*self.v[i]+(1-self.beta2)*p.g**2
            p.kha += -self.eta*np.sqrt(1-self.beta2**self.t)/(1-self.beta1**self.t)*self.m[i]/np.sqrt(self.v[i])
            self.t += 1
            p.g = 0



class Dropout(Chan):
    def __init__(self,drop=0.5):
        self.drop = drop
        self.fuekyu = 1
        
    def pai(self,x):
        if(self.fuekyu):
            self.krong = np.random.rand(*x.shape)>self.drop
            return x*self.krong
        else:
            return x*(1.-self.drop)
    
    def yon(self,g):
        return g*self.krong

class Batchnorm(Chan):
    def __init__(self,m,mmt=0.9):
        self.m = m
        self.param = [Param(np.ones(m)),Param(np.zeros(m))]
        self.rmu = np.zeros(m)
        self.rvar = np.zeros(m)+1e-8
        self.mmt = mmt
        self.fuekyu = 1
    
    def pai(self,x):
        if(self.fuekyu):
            self.n = len(x)
            mu = x.mean(0)
            self.xc = x-mu
            var = (self.xc**2).mean(0)+1e-8
            self.sigma = np.sqrt(var)
            self.xn = xn = self.xc/self.sigma
            self.rmu = self.mmt*self.rmu + (1.-self.mmt)*mu
            self.rvar = self.mmt*self.rvar + (1.-self.mmt)*var
        else:
            xc = x - self.rmu
            xn = xc/np.sqrt(self.rvar)
        
        return self.param[0].kha*xn+self.param[1].kha
    
    def yon(self,g):
        self.param[0].g = (g*self.xn).sum(0)
        self.param[1].g = g.sum(0)
        gxn = self.param[0].kha*g
        gsigma = -((gxn*self.xc)/self.sigma**2).sum(0)
        gvar = gsigma/self.sigma/2
        gxc = gxn/self.sigma + (2./self.n)*self.xc*gvar
        gmu = gxc.sum(0)
        gx = gxc - gmu/self.n
        return gx



class Lrelu(Chan):
    def __init__(self,a=0.01):
        self.a = a
    
    def pai(self,x):
        self.krong = (x>0)
        return x*np.where(self.krong,1,self.a)
    
    def yon(self,g):
        return g*np.where(self.krong,1,self.a)

class Prelu(Chan):
    def __init__(self,m,a=0.25):
        self.param = [Param(np.ones(m)*a)]
    
    def pai(self,x):
        self.krong = (x>0)
        self.x = x
        return x*np.where(self.krong,1,self.param[0].kha)
    
    def yon(self,g):
        self.param[0].g += (self.x*(self.krong==0)).sum(0)
        return g*np.where(self.krong,1,self.param[0].kha)

class Elu(Chan):
    def __init__(self,a=1):
        self.a = a
    
    def pai(self,x):
        self.krong = (x>0)
        self.h = np.where(self.krong,x,self.a*(np.exp(x)-1))
        return self.h
    
    def yon(self,g):
        return g*np.where(self.krong,1,(self.h+self.a))

class Selu(Chan):
    a = 1.6732632423543772848170429916717
    l = 1.0507009873554804934193349852946
    
    def pai(self,x):
        self.krong = (x>0)
        self.h = self.l*np.where(self.krong,x,self.a*(np.exp(x)-1))
        return self.h
    
    def yon(self,g):
        return g*self.l*np.where(self.krong,1,(self.h+self.a))

class Tanh(Chan):
    def pai(self,x):
        self.h = np.tanh(x)
        return self.h
    
    def yon(self,g):
        return g*(1-self.h**2)

class Softsign(Chan):
    def pai(self,x):
        self.abs_x_1 = np.abs(x)+1
        return x/self.abs_x_1
    
    def yon(self,g):
        return g/self.abs_x_1**2

class Softplus(Chan):
    def pai(self,x):
        self.exp_x = np.exp(x)
        return np.log(1+self.exp_x)
    
    def yon(self,g):
        return g*self.exp_x/(1+self.exp_x)
    


class Conv1d(Chan):
    def __init__(self,m0,m1,kk,st=1,pad=0,sigma=1):
        self.param = [Param(np.random.normal(0,sigma,[m1,m0,kk])),
                      Param(np.zeros(m1))]
        self.st = st
        self.pad = pad
    
    def pai(self,X):
        X = np.pad(X,[(0,0),(0,0),(self.pad,self.pad)],'constant')
        m1,m0,kk = self.param[0].kha.shape
        n,m0_,k0 = X.shape
        assert m0_==m0
        k1 = int((k0-kk)/self.st)+1
        X_ = np.zeros([n,m0,kk,k1])
        for i in range(kk):
            X_[:,:,i,:] = X[:,:,i:i+k1*self.st:self.st]
        X_ = X_.transpose(0,3,1,2).reshape(-1,m0*kk)
        w = self.param[0].kha.reshape(m1,-1).T
        b = self.param[1].kha
        a = np.dot(X_,w) + b
        a = a.reshape(n,k1,-1).transpose(0,2,1)
        self.ruprang = n,m1,m0,kk,k0,k1
        self.X_ = X_
        return a
    
    def yon(self,g):
        n,m1,m0,kk,k0,k1 = self.ruprang
        g = g.transpose(0,2,1).reshape(-1,m1)
        w = self.param[0].kha.reshape(m1,-1).T
        self.param[0].g = np.dot(self.X_.T,g).transpose(1,0).reshape(m1,m0,kk)
        self.param[1].g = g.sum(0)
        gX_ = np.dot(g,w.T)
        gX_ = gX_.reshape(-1,k1,m0,kk).transpose(0,2,3,1)
        gX = np.zeros([n,m0,k0+self.pad*2])
        for i in range(kk):
            gX[:,:,i:i+k1*self.st:self.st] += gX_[:,:,i,:]
        return gX[:,:,self.pad:k0-self.pad]

class Conv2d(Chan):
    def __init__(self,m0,m1,kk,st=1,pad=0,sigma=1):
        if(type(kk)==int):
            kk = [kk,kk]
        if(type(st)==int):
            st = [st,st]
        if(type(pad)==int):
            pad = [pad,pad]
        
        self.param = [Param(np.random.normal(0,sigma,[m1,m0,kk[1],kk[0]])),
                      Param(np.zeros(m1))]
        self.st = st
        self.pad = pad
    
    def pai(self,X):
        px,py = self.pad
        stx,sty = self.st
        X = np.pad(X,[(0,0),(0,0),(py,py),(px,px)],'constant')
        m1,m0,kky,kkx = self.param[0].kha.shape
        n,m0_,ky0,kx0 = X.shape
        assert m0_==m0
        kx1 = int((kx0-kkx)/stx)+1
        ky1 = int((ky0-kky)/sty)+1
        X_ = np.zeros([n,m0,kky,kkx,ky1,kx1])
        for _j in range(kky):
            j_ = _j+ky1*sty
            for _i in range(kkx):
                i_ = _i+kx1*stx
                X_[:,:,_j,_i,:,:] = X[:,:,_j:j_:sty,_i:i_:stx]
        X_ = X_.transpose(0,4,5,1,2,3).reshape(-1,m0*kkx*kky)
        w = self.param[0].kha.reshape(m1,-1).T
        b = self.param[1].kha
        a = np.dot(X_,w) + b
        a = a.reshape(n,ky1,kx1,-1).transpose(0,3,1,2)
        self.ruprang = n,m1,m0,kky,kkx,ky0,kx0,ky1,kx1
        self.X_ = X_
        return a
    
    def yon(self,g):
        px,py = self.pad
        stx,sty = self.st
        n,m1,m0,kky,kkx,ky0,kx0,ky1,kx1 = self.ruprang
        g = g.transpose(0,2,3,1).reshape(-1,m1)
        w = self.param[0].kha.reshape(m1,-1).T
        self.param[0].g = np.dot(self.X_.T,g).transpose(1,0).reshape(m1,m0,kky,kkx)
        self.param[1].g = g.sum(0)
        gX_ = np.dot(g,w.T)
        gX_ = gX_.reshape(-1,ky1,kx1,m0,kky,kkx).transpose(0,3,4,5,1,2)
        gX = np.zeros([n,m0,ky0+2*py,kx0+2*px])
        for _j in range(kky):
            j_ = _j+ky1*sty
            for _i in range(kkx):
                i_ = _i+kx1*stx
                gX[:,:,_j:j_:sty,_i:i_:stx] += gX_[:,:,_j,_i,:,:]
        return gX[:,:,py:ky0-py,px:kx0-px]

class MaxP1d(Chan):
    def __init__(self,kk,st=None):
        self.kk = kk
        if(st==None):
            self.st = self.kk
        else:
            self.st = st
    
    def pai(self,X):
        n,m,k0 = X.shape
        k1 = int((k0-self.kk)/self.st)+1
        X_ = np.zeros([n,m,self.kk,k1])
        for i in range(self.kk):
            X_[:,:,i,:] = X[:,:,i:i+k1*self.st:self.st]
        X_ = X_.transpose(0,3,1,2).reshape(-1,self.kk)
        self.argmax = X_.argmax(1)
        self.ruprang = n,m,k0,k1
        return X_.max(1).reshape(n,k1,m).transpose(0,2,1)
    
    def yon(self,g):
        g = g.transpose(0,2,1)
        n,m,k0,k1 = self.ruprang
        gX_ = np.zeros([g.size,self.kk])
        gX_[np.arange(len(self.argmax)),self.argmax] = g.flatten()
        gX_ = gX_.reshape(-1,k1,m,self.kk).transpose(0,2,3,1)
        gX = np.zeros([n,m,k0])
        for i in range(self.kk):
            gX[:,:,i:i+k1*self.st:self.st] += gX_[:,:,i,:]
        return gX

class MaxP2d(Chan):
    def __init__(self,kk,st=None):
        if(type(kk)==int):
            self.kk = [kk,kk]
        else:
            self.kk = kk
        
        if(st==None):
            self.st = self.kk
        elif(type(st)==int):
            self.st = [st,st]
        else:
            self.st = st
    
    def pai(self,X):
        stx,sty = self.st
        kkx,kky = self.kk
        n,m,ky0,kx0 = X.shape
        kx1 = int((kx0-kkx)/stx)+1
        ky1 = int((ky0-kky)/sty)+1
        X_ = np.zeros([n,m,kky,kkx,ky1,kx1])
        for _j in range(kky):
            j_ = _j+ky1*sty
            for _i in range(kkx):
                i_ = _i+kx1*stx
                X_[:,:,_j,_i,:,:] = X[:,:,_j:j_:sty,_i:i_:stx]
        X_ = X_.transpose(0,4,5,1,2,3).reshape(-1,kkx*kky)
        self.argmax = X_.argmax(1)
        self.ruprang = n,m,ky0,kx0,ky1,kx1
        return X_.max(1).reshape(n,ky1,kx1,m).transpose(0,3,1,2)
    
    def yon(self,g):
        g = g.transpose(0,2,3,1)
        stx,sty = self.st
        kkx,kky = self.kk
        n,m,ky0,kx0,ky1,kx1 = self.ruprang
        gX_ = np.zeros([g.size,kkx*kky])
        gX_[np.arange(self.argmax.size),self.argmax.flatten()] = g.flatten()
        gX_ = gX_.reshape(-1,ky1,kx1,m,kky,kkx).transpose(0,3,4,5,1,2)
        gX = np.zeros([n,m,ky0,kx0])
        for _j in range(kky):
            j_ = _j+ky1*sty
            for _i in range(kkx):
                i_ = _i+kx1*stx
                gX[:,:,_j:j_:sty,_i:i_:stx] += gX_[:,:,_j,_i,:,:]
        return gX

class Plianrup(Chan):
    def __init__(self,*rupmai):
        self.rupmai = rupmai
        
    def pai(self,x):
        self.rupdoem = x.shape
        return x.reshape(self.rupdoem[0],*self.rupmai)
    
    def yon(self,g):
        return g.reshape(*self.rupdoem)



def sigmoid(x): return Sigmoid()(x)
def relu(x): return Relu()(x)
def lrelu(x): return Lrelu()(x)
def elu(x): return Elu()(x)
def selu(x): return Selu()(x)
def tanh(x): return Tanh()(x)
def softplus(x): return Softplus()(x)
def softsign(x): return Softsign()(x)
def softmax_entropy(a,Z): return Softmax_entropy()(a,Z)
def sigmoid_entropy(a,z): return Sigmoid_entropy()(a,z)
def mse(h,z): return Mse()(h,z)