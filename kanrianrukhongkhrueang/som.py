import numpy as np

class SOM:
    def __init__(self,ruprang=(20,20),eta=0.1):
        if(type(ruprang)==int):
            self.ruprang = [ruprang]
        else:
            self.ruprang = ruprang
        self.eta = eta
        self.miti = len(ruprang)
        self.r = max(self.ruprang)
    
    def rianru(self,X,n_thamsam=100):
        self.tau = n_thamsam/np.log(self.r)
        w = [np.random.uniform(X[:,i].min(),X[:,i].max(),np.prod(self.ruprang)) for i in range(X.shape[1])]
        self.w = w = np.array(w).T
        c = np.meshgrid(*[np.arange(i) for i in self.ruprang],indexing='ij')
        self.c = c = np.stack(c,-1).reshape(-1,self.miti)
        for t in range(n_thamsam):
            e = np.exp(-t/self.tau)
            self.r_t = self.r*e
            self.eta_t = self.eta*e
            for x in np.random.permutation(X):
                x_w = x-w
                i = np.argmin((x_w**2).sum(1))
                c_klaisut = self.c[i]
                d2 = ((self.c-c_klaisut)**2).sum(1)
                f = np.exp(-0.5*d2/self.r_t**2)
                w += (self.eta_t*f)[:,None]*x_w
    
    def plaeng(self,X,ao_rayahang=0):
        if(X.ndim==1):
            x_w = X-self.w
            i = np.argmin((x_w**2).sum(1))
            if(ao_rayahang):
                return self.c[i],np.sqrt((x_w[i]**2).sum())
            else:
                return self.c[i]
        else:
            if(ao_rayahang):
                w_klai = []
                rayahang2 = []
                for x_w_2 in ((X[:,None]-self.w)**2).sum(2):
                    i = np.argmin(x_w_2)
                    w_klai.append(self.c[i])
                    rayahang2.append(x_w_2[i])
                rayahang = np.sqrt(np.array(rayahang2))
                return np.array(w_klai),rayahang
            else:
                return np.array([self.c[np.argmin(x_w_2)] for x_w_2 in ((X[:,None]-self.w)**2).sum(2)])
        
    def plaengklap(self,c=None):
        w = self.w.reshape(list(self.ruprang)+[-1])
        if(c==None):
            return w.transpose(np.arange(self.miti-1,-2,-1))
        elif(c.ndim==1):
            return w[tuple(c)]
        else:
            return np.array([w[tuple(ci)] for ci in c])
    
    def rianru_plaeng(self,X,n_thamsam=100,ao_rayahang=0):
        self.rianru(X,n_thamsam)
        return self.plaeng(X,ao_rayahang)