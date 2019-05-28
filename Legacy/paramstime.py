"""
u=[]
b=False
for i in t:
    if i[-2]!=']':
        ii=i[:-1]
        b=True
        u.append(ii)
    else:
        if b:
            u[-1]=u[-1]+i
            b=False
        else :
            u.append(i)

v=[]
for i in u:
    ii=list(i)
    k=1
    while k < len(ii)-1:
        if ii[k]==' ':
            while ii[k+1]==' ':
                _=ii.pop(k+1)
            ii[k]=','
        if ii[k]==',' and ii[k-1]=='[':
            _=ii.pop(k)
        else:
            k+=1
    j=''
    for n in ii:
        j=j+n
    v.append(j[:-1])

lala=[eval(i) for i in v]
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rc('font', size=9)
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)

t=np.load('results/paramsevolutionnorm.npy')
"""t[:,0]=t[:,0]/t[:,0][-1]
t[:,1]=t[:,1]/t[:,1][-1]
t[:,4]=t[:,4]/t[:,4][-1]
t[:,5]=t[:,5]/t[:,5][-1]
t[:,2]=(t[:,2]%np.pi)/np.pi
t[:,3]=(t[:,3]%np.pi)/np.pi
"""
t=t/t[-1]
"""
tb=[]
for i in range(20):
    x=[]
    for j in range(6):
        x.append(1+0.01*np.random.uniform(-1,1)/(i+1))
    tb.append(x)

t=np.concatenate((t,np.array(tb)))
"""

fig,axes=plt.subplots(2,3)
fig.set_size_inches(8,5)
for i in range(6):
    ax=plt.subplot(2,3,i+1)
    plt.plot(t[:,i])
    plt.grid(True)
    if i == 3:
        ax.set_xlabel('Function calls')
        ax.set_ylabel('Value (normalized)')


plt.suptitle('Parameters optimization during a minimization',size=14)
plt.savefig('evolutionplot.png',dpi=600)
