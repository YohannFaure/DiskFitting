#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
This is designed to make some plots with an already generated samplers list.
'''

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import corner
#from ModelingEmcee import *
##### Open the data
location = '/home/yohann/Desktop/Stage2019/DiskFitting/results/optimization/opti_37_300_1000part2.npy'
samples,thetaminbis,thetamaxbis,labels = np.load(location)
# thetamin and thetamax are defined in the ModelingEmcee.py file. It correspnds to the limits of the fitting. Labels are just the names of the parameters

##### Get the shape of the plot
nwalkers,iterations,ndims = samples.shape
ncols = 4
nrows =math.ceil( ndims / ncols )

##### Make a figure
fig, axes = plt.subplots(nrows=nrows,ncols=ncols, figsize=(20, 25), sharex=True)
for i in range(ndims):
    ax = axes.flatten()[i]
    _=ax.plot(np.transpose(samples[:, :, i]), "k", alpha=0.3)
    _=ax.set_xlim(0, iterations)
    _=ax.set_ylabel(labels[i])
#    _=ax.yaxis.set_label_coords(-0.1, 0.5)
    _=ax.plot([0,iterations],[thetaminbis[i],thetaminbis[i]])
    _=ax.plot([0,iterations],[thetamaxbis[i],thetamaxbis[i]])

_=ax.set_xlabel('iterations')
plt.tight_layout()
plt.show()

##### This would be for a cornerplot
figure = corner.corner(samples[:,-1,:3])
plt.show()


