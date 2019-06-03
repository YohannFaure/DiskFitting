#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
This is designed to make some plots with an already generated samplers list.

usage example : 
python3 PlotEmcee.py results/optimization/opti_37_300_1000part5.npy --save hahaha --size '(11.25,15)'
'''

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import corner
import matplotlib as mpl

"""
mpl.use('pgf')
pgf_with_rc_fonts = {
    "font.family": "serif",
    "font.serif": [],                   # use latex default serif font
    "font.sans-serif": [], # use a specific sans-serif font
    "font.size": 8}

mpl.rcParams.update(pgf_with_rc_fonts)
plt.style.use('dark_background')
"""


def multiplot(samples,figshape=None,size=(20,25),save=None,limits=None):
    """Plots the parameters in function of time
    """
    ##### get info
    samples=samples[:,::10,:]
    nwalkers,iterations,ndims = samples.shape
    ##### set size of the figure
    if figshape:
        ncols,nrow=figshape
    else:
        ncols = 4
        nrows =math.ceil( ndims / ncols )
    ##### plot
    fig, axes = plt.subplots(nrows=nrows,ncols=ncols, figsize=size, sharex=True)
    for i in range(ndims):
        ax = axes.flatten()[i]
        _=ax.plot(np.transpose(samples[:, :, i]), "k", alpha=0.2)
        _=ax.set_xlim(0, iterations)
        _=ax.set_ylabel(labels[i])
        ax.grid(True)
    #    _=ax.yaxis.set_label_coords(-0.1, 0.5)
        if limits:
            thetaminbis,thetamaxbis=limits
            _=ax.plot([0,iterations],[thetaminbis[i],thetaminbis[i]])
            _=ax.plot([0,iterations],[thetamaxbis[i],thetamaxbis[i]])
    _=ax.set_xlabel('iterations')
    plt.tight_layout()
    ##### save
    if save:
        plt.savefig('results/{}_multiplot.pgf'.format(save))
    else:
        plt.show()
    return(None)

def cornerplot(samples,labels,save=None):
    """Just makes a cornerplot, but makes it easier
    Beware of the segmentation faults..."""
    nwalkers,iterations,ndims = samples.shape
    ndims=min(6,ndims)
    cornering=(samples[:,-iterations//2:,:ndims].reshape((-1,ndims)))
    fig = corner.corner(cornering, quantiles=[0.16, 0.50, 0.84],labels=labels[:ndims],show_titles=True,label_kwargs={'labelpad':20, 'fontsize':0}, fontsize=8)
    fig.set_size_inches(6,3.2)
    if save:
        fig.savefig("results/{}_cornerplot.pgf".format(save))
    else:
        fig.show()
    return(None)


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description = 'arguments')
    parser.add_argument("location", help = "File to use as input.",type = str)
    parser.add_argument("--figshape", help = "Shape of the multiple plot.",type = str,default="None")
    parser.add_argument("--size", help = "Size of the multiple plot.",type = str,default="(20,25)")
    parser.add_argument("--save", help = "Will not plot the result and only give it if present.",type = str, default=None)
    args = parser.parse_args()
    ##### Open the data
    #location = '/home/yohann/Desktop/Stage2019/DiskFitting/results/optimization/opti_37_300_5000part3.npy'
    samples,thetaminbis,thetamaxbis,labels = np.load(args.location)
    # thetamin and thetamax are defined in the ModelingEmcee.py file. It correspnds to the limits of the fitting. Labels are just the names of the parameters
    size=eval(args.size)
    figshape=eval(args.figshape)
    if args.save :
        save = args.save
    else:
        save = None
    multiplot(samples,figshape=figshape,size=size,save=save,limits=(thetaminbis,thetamaxbis))
    #cornerplot(samples,labels,save=save)

