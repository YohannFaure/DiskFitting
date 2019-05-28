#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from FunctionsModule import *

inc,pa=4.59835388e+01,-3.42477693e+01
location='J1615_LB_ap_0.5.fits'

def OneGaussianOneRing(x0,y0,inc,pa,A0,sigma0,A1,r1,sigma1,mesh):
    return(gaussianfit((x0,y0,inc,pa,A0,sigma0), mesh)+radialgaussian((A1,r1,sigma1,inc,pa),mesh))

def FitThatHell(location,inc,pa,cmap='inferno',center=None,width=None,method='Powell'):
    """lololo
    """
    ##### Opening the image properly
    inc,pa=degtorad*inc,degtorad*pa
    header,image=openimage(location)
    lx, ly = image.shape
    pixtosec=np.abs(3600*header['CDELT1'])
    if center:
        xc,yc=center
    else:
        xc,yc=lx//2,ly//2
    if width:
        size=width//2
    else:
        size=min(lx,ly)//2,
    ##### Resizing it
    image=resizedimage(image,xc,yc,size)
    llx,lly=image.shape
    x0,y0=llx/2,lly/2
    headerscale=(llx/lx,lly/ly) #keep the scale for plotting
    maximage=np.max(image)
    image=image/np.max(image) #normalize image, to get better results with the minimization
    seed=[1,llx/10,1,llx/10,llx/100,2,1,1,llx/5,llx/10]
    mesh=meshmaker(llx,lly,llx/2,lly/2)
    def Minimize2(params):
        print(params)
        A0,sigma0,A1,r1,sigma1,A2,sigma2,A3,r3,sigma3=params
        model = gaussianfit((x0,y0,inc,pa,A0,sigma0), mesh)+gaussianfit((x0,y0,inc,pa,A2,sigma2), mesh)+radialgaussian((A1,r1,sigma1,inc,pa),mesh)+radialgaussian((A3,r3,sigma3,inc,pa),mesh)
        return(quadcost(image,model))
    print('\n\nWorking on it\n\n')
    test = opt.minimize(Minimize2,seed,method=method)
    A0,sigma0,A1,r1,sigma1,A2,sigma2,A3,r3,sigma3=test['x']
    model = gaussianfit((x0,y0,inc,pa,A0,sigma0), mesh)+gaussianfit((x0,y0,inc,pa,A2,sigma2), mesh)+radialgaussian((A1,r1,sigma1,inc,pa),mesh)+radialgaussian((A3,r3,sigma3,inc,pa),mesh)
    return(test,model)

t,m=FitThatHell(location,inc,pa,width=500)

