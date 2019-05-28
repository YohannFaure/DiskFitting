import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import corner


location = '/home/yohann/Desktop/Stage2019/code/results/optimization/opti_37_100_200.npy'

samples,thetamin,thetamax,labels = np.load(location)

nwalkers,iterations,ndims = samples.shape
ncols = 4
nrows =math.ceil( ndims / ncols )

fig, axes = plt.subplots(nrows=nrows,ncols=ncols, figsize=(20, 25), sharex=True)

for i in range(ndims):
    ax = axes.flatten()[i]
    _=ax.plot(np.transpose(samples[:, :, i]), "k", alpha=0.3)
    _=ax.set_xlim(0, iterations)
    _=ax.set_ylabel(labels[i])
#    _=ax.yaxis.set_label_coords(-0.1, 0.5)
#    _=ax.plot([0,iterations],[thetaminbis[i],thetaminbis[i]])
#    _=ax.plot([0,iterations],[thetamaxbis[i],thetamaxbis[i]])
_=ax.set_xlabel('iterations')
plt.tight_layout()
plt.show()

figure = corner.corner(samples[:,-1,:3])
plt.show()


