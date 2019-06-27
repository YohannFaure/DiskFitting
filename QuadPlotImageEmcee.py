#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plots an image for the emcee optimization.
Quite custom, you might want to edit it.
"""

from PlotEmcee import *
from OptimizationModule import *
import matplotlib as mpl

emceelocation='results/optimization/opti_37_300_1000part30.npy'

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

##### Define the model used
ring1=FFRing()
ring2=GaussianRing()
ring3=GaussianRing()
ring4=GaussianRing()
centralgauss=Gaussian2D()
model=centralgauss+ring1+ring2+ring3+ring4


samples,_,_,_=np.load(emceelocation,allow_pickle=True)
theta = ExtractPercentiles(samples)[1]

model.parameters=theta

def modelplot2(model,image,cmap,header,size=(6,3.5),save='Quadplot2.pgf'):#time())):
    """
    Plots a quadri-images graph, need to be adapted to your own plot, as it is higgly custom...
    """
    ##### settings
#    if save!='False':
#        mpl.use('pgf')
#        pgf_with_rc_fonts = {
#            "font.family": "serif",
#            "font.serif": [],                   # use latex default serif font
#            "font.sans-serif": [], # use a specific sans-serif font
#            "font.size": 8}
#        plt.style.use('seaborn-whitegrid')
#        mpl.rcParams.update(pgf_with_rc_fonts)
    maximage=np.max(image)
    image,model=image/maximage,model/maximage
    fig,axarr=plt.subplots(2,2)
    #plt.suptitle('Fitting of a brightness profile\n(normalization: {:5.5f} {})'.format(maximage,header['BUNIT']),fontsize=11)
    fig.set_size_inches(size[0],size[1])
    ##### scale for pixel--arcsec conversion
    par=header["PAR"]
    pixtosec=np.abs(3600*header['CDELT1'])
    pixtoau=1000*pixtosec/(par)
    arcsectoau=1000/par
    scalex=image.shape[0]*pixtoau/2
    scaley=image.shape[1]*pixtoau/2
    ##### setting and plotting each subplots
    ### Titles
    axarr[0,0].title.set_text(r'Fitting model')
    axarr[0,1].title.set_text(r'10$\times$Residual')
    axarr[1,0].title.set_text(r'Original image')
    axarr[1,1].title.set_text(r'Residual\\absolute value')
    ### Plots
    axarr[0,0].imshow(model, vmin=0, vmax=1, interpolation='nearest',origin=0,cmap=cmap,extent=[scalex,-scalex,-scaley, scaley])
    axarr[0,1].imshow(10*np.abs(model-image), vmin=0, vmax=1, interpolation='nearest',origin=0,cmap=cmap,extent=[scalex,-scalex,-scaley, scaley])
    axarr[1,0].imshow(image, vmin=0, vmax=1, interpolation='nearest',origin=0,cmap=cmap,extent=[scalex,-scalex,-scaley, scaley])
    ### resolution bean on the 3rd subplot
    beam = mpl.patches.Ellipse(xy=(0.9*scalex,-0.9*scalex), width=header['BMIN']*3600*arcsectoau, height=header['BMAJ']*3600*arcsectoau , color='white', fill=True, angle=header['BPA'])
    axarr[1,0].add_artist(beam)
    ### legend
    axarr[1,0].set_xlabel('East-West (au)')
    axarr[1,0].set_ylabel('South-North (au)')
#    axarr[1,1].set_xlabel('Inclination: {inc:3.2f}. Position angle: {pa:3.2f}. (deg)'.format(inc=params[2]*radtodeg,pa=params[3]*radtodeg))
    ### last subplot
    im=axarr[1,1].imshow(np.absolute(image - model), vmin=0, vmax=1, interpolation='nearest',origin=0,cmap=cmap,extent=[scalex,-scalex,-scaley, scaley])
    ##### layout and colorbar
    plt.tight_layout()
    #plt.grid()
    plt.subplots_adjust(top=0.85)
    cax = plt.axes([0.93, 0.12, 0.015, 0.7])
    plt.colorbar(im,cax=cax)
    plt.savefig('results/QuadplotEmcee.pdf')
    return(None)


modelplot2(model(xx,yy),image,'inferno',header)
