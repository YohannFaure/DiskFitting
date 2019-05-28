#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
This program is a training exemple of data fitting for protoplanetary disks.
It takes an image (location) and finds it's

Yohann Faure
yohann.faure@ens-lyon.fr
'''
from FunctionsModule import *

##### Options
plt.rc('font', size=9)
cmap = 'inferno'
location = 'J1615_LB_ap_0.5.fits'

"""
##### Physical parameters of my observation
inc = 47. * np.pi / 180.            #inclination
pa =  145.75 * np.pi / 180.         #pa
paralax = 6.3417086075333895        #mas
dist = 1. / (paralax * 10**-3)      #parsec
"""


##############################################################################
#                                 DATA READ                                  #
##############################################################################



##### Opening the image properly
header,image=openimage(location)
l1, l2 = image.shape
xc,yc=l1//2,l2//2
size=700                            #To be set to whatever is adapted
image=resizedimage(image,xc,yc,700)
#ra,dec,ext=extractheaderinfo(header)

##### Defining the mesh
l1,l2=image.shape
x = np.arange(l1)
y = np.arange(l2)
x, y = np.meshgrid(x, y)
mesh=x,y


##############################################################################
#                           Fitting with a gaussian                          #
##############################################################################

'''
Fit using minimization of chi2 in the image plane

SIMPLE GAUSSIAN
'''

image=image/np.max(image)
seed = np.array([700,700,0,0,1.,100.])
blur = blurimage(image,sigma = 20)

def cost(img,model):
    return(np.sum((img-model)**2))

def functiontominimize(params):
    model = gaussianfit(params, mesh)
    return(cost(blur,model))




#methods = ['Nelder-Mead','Powell','CG','BFGS']
#i=0 nope
#i=1 ok 41s
#i=2 ok 318s
#i=3 ~ok 46s

test = opt.minimize(functiontominimize,seed,method='Powell')

params = test['x']
model = gaussianfit(params, mesh)
fig,axarr=plt.subplots(2,2)
axarr[0,0].imshow(model, interpolation='nearest',origin=0,cmap=cmap)
axarr[0,1].imshow(blur, interpolation='nearest',origin=0,cmap=cmap)
axarr[1,0].imshow(image, interpolation='nearest',origin=0,cmap=cmap)
axarr[1,1].imshow(np.absolute(image - model), interpolation='nearest',origin=0,cmap=cmap)

axarr[0,0].title.set_text('Gaussian feat')
axarr[0,1].title.set_text('Blured image')
axarr[1,0].title.set_text('Original image')
axarr[1,1].title.set_text('Residual absolute value')

plt.show()
print(params)
