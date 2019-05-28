#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Lets have fun !
"""
from FunctionsModule import *
from TiltFinder import *
from astropy.modeling import models, fitting
from astropy.convolution import convolve_models


import timeit


fit_LSQ=fitting.LevMarLSQFitter()
##### Finding tilt and real coordinates
location="J1615_edit.fits"
tilt=TiltFinder(location,center=(1500,1500),width=600,noplot=True)
xc,yc,inc,pa,_,_=tilt
header,image=openimage(location)
image=resizedimage(image,1500,1500,400)
x,y=meshmaker(800,800,xc+100,yc+100)
xx,yy,rr,angles=deprojectedcoordinates(x,y,inc,pa)
pixtosec,pixtoau,arcsectoau=GlobalVars(header)

##### Define the beam convolution
beamgauss = models.Gaussian2D(amplitude=1, x_mean=0., y_mean=0., x_stddev=header['BMIN']*3600/pixtosec, y_stddev=header['BMAJ']*3600/pixtosec, theta=header['BPA']*degtorad)
# beware, the theta argument of this function seems like shit.


"""

##### Having fun with fitting things from astropy


##### First model :
centralgaussianinit=models.Gaussian2D(amplitude=0.00306785, x_mean=-0.01943452, y_mean=0.07688838, x_stddev=121.85977989, y_stddev=125.12070432, theta=0.07164449)

CentralGaussian=fit_LSQ(centralgaussianinit,xx,yy,image)


##### Second model :
centralmoffatinit=models.Moffat2D(amplitude=0.00404319, x_0=0.22794102, y_0=-0.22554013, gamma=113.44560995, alpha=1.19040464)
fit_LSQ=fitting.LevMarLSQFitter()
centralmoffat=fit_LSQ(centralmoffatinit,xx,yy,image)

##### third model :  !! SHIT
centralmexinit=models.MexicanHat2D(amplitude=0.00003816, x_0=-0.11055477, y_0=-0.3114417, sigma=0.01017698)
fit_LSQ=fitting.LevMarLSQFitter()
centralmex=fit_LSQ(centralmexinit,xx,yy,image)

#>>> quadcost(image,CentralGaussian(xx,yy))
#0.025242072181772362
#>>> quadcost(image,centralmoffat(xx,yy))
#0.01689292190601885
#>>> quadcost(image,centralmex(xx,yy))
#0.340814910270637


### One gaussian + moffat :
fit_LSQ=fitting.LevMarLSQFitter()
GaussInit=models.Gaussian2D(amplitude=0.00306785, x_mean=-0.01943452, y_mean=0.07688838, x_stddev=121.85977989, y_stddev=125.12070432, theta=0.07164449)
#moffatinit = models.Moffat2D(amplitude=0.00404319, x_0=0.22794102, y_0=-0.22554013, gamma=113.44560995, alpha=1.19040464)



##### But one can create his own model !!
@models.custom_model
def GaussianRing(x, y, amplitude=1., xc=0., yc=0., width=1., a=1.,b=1.,theta=0.,cut=1.):
    c=np.cos(theta)
    s=np.sin(theta)
    xx=x-xc
    yy=y-yc
    u=np.sqrt(((xx*c+yy*s)/a)**2+((yy*c-xx*s)/b)**2) - 1
    ff=np.sqrt(a*b) #width normalization
    gauss = np.exp(-(u**2)/(2*(width/ff)**2))
    amp = np.maximum(gauss,np.ones(gauss.shape)*(cut))
    return(gauss*amplitude)

ring1=GaussianRing(amplitude=0.003,width=2,a=50,b=50)
ring2=GaussianRing(amplitude=0.003,width=30,a=70,b=70,cut=0.5)
centralgauss=models.Gaussian2D(amplitude=0.05, x_mean=0., y_mean=0., x_stddev=5., y_stddev=5., theta=0.)

showi(ring2(xx,yy))
#model=convolve_models(ring+centralgauss,beamgauss)
#showi(model(xx,yy))

model=ring1+ring2+centralgauss

MyEnd = fit_LSQ(model,xx,yy,image)


##### Other test

@models.custom_model
def LinearRing(x, y, amplitude=1., xc=0., yc=0., a=1.,b=1.,theta=0.,end=200.):
    c=np.cos(theta)
    s=np.sin(theta)
    xx=x-xc
    yy=y-yc
    u=np.sqrt(((xx*c+yy*s)/a)**2+((yy*c-xx*s)/b)**2) - 1
    end=end/np.sqrt(a*b)
    interbool=(u>0)
    boolout=interbool*(u<end)
    ampout = boolout*(end-u)/end
    ampin = (1-interbool)*(-np.min(u)+u)/(-np.min(u))
    amp = ampout+ampin
    return(amp*(amplitude))


ring1=LinearRing(amplitude=0.0036,a=50.,b=50.,end=190.)
ring2=GaussianRing(amplitude=0.0005,a=250.,b=210.,width=20.,cut=1.)
ring3=GaussianRing(amplitude=0.0004,a=270.,b=270.,width=20.,cut=1.)
centralgauss=models.Gaussian2D(amplitude=0.0035, x_mean=0., y_mean=0., x_stddev=5., y_stddev=5., theta=0.)


def tieab(g):
    return(g.b)


ring1.theta.fixed=True
ring2.theta.fixed=True
ring3.theta.fixed=True
ring1.end.fixed=True
centralgauss.theta.fixed=True
#ring1.a.tied = tieab
#ring2.a.tied = tieab
#centralgauss.a.tied = tieab

model=centralgauss+ring1+ring2+ring3

modelend=fit_LSQ(model,xx,yy,image)
showi(modelend(xx,yy))

modelend.theta_0.fixed=False
modelend.theta_2.fixed=False
modelend.end_1.fixed=False
modelend.theta_1.fixed=False

modelend=fit_LSQ(modelend,xx,yy,image)
showi(modelend(xx,yy))


##### Other test
"""

@models.custom_model
def GaussianRing(x, y, amplitude=1., xc=0., yc=0., width=1., a=1.,b=1.,theta=0.,cut=1.):
    c=np.cos(theta)
    s=np.sin(theta)
    xx=x-xc
    yy=y-yc
    u=np.sqrt(((xx*c+yy*s)/a)**2+((yy*c-xx*s)/b)**2) - 1
    ff=np.sqrt(a*b) #width normalization
    gauss = np.exp(-(u**2)/(2*(width/ff)**2))
    amp = np.maximum(gauss,np.ones(gauss.shape)*(cut))
    return(gauss*amplitude)

@models.custom_model
def LinearRing(x, y, amplitude=1., xc=0., yc=0., a=1.,b=1.,theta=0.,end=200.):
    c=np.cos(theta)
    s=np.sin(theta)
    xx=x-xc
    yy=y-yc
    u=np.sqrt(((xx*c+yy*s)/a)**2+((yy*c-xx*s)/b)**2) - 1
    end=end/np.sqrt(a*b)
    interbool=(u>0)
    boolout=interbool*(u<end)
    ampout = boolout*(end-u)/end
    ampin = (1-interbool)*(-np.min(u)+u)/(-np.min(u))
    amp = ampout+ampin
    return(amp*(amplitude))

@models.custom_model
def FFRing(x,y,i0=1.,i1=1.,sig0=1.,sig1=1.,gam=1.,xc=0.,yc=0.,a=1.,b=1.,theta=0.):#Facchini
    c=np.cos(theta)
    s=np.sin(theta)
    xx=x-xc
    yy=y-yc
    u=np.sqrt(((xx*c+yy*s)/a)**2+((yy*c-xx*s)/b)**2)
    f = i0 * ((u / sig0)**gam) * np.exp(-(u**2) / (2 * sig0**2)) 
    g = i1 * np.exp(-(u**2) / (2 * sig1**2))
    return(f+g)


ring1=FFRing(i0=0.0036,i1=0.0036,sig0=1.,sig1=0.1,gam=1.6,xc=0.,yc=0.,a=50.,b=50.,theta=0.)
ring2=GaussianRing(amplitude=0.0005,a=230.,b=230.,width=20.,cut=1.)
ring3=GaussianRing(amplitude=0.0004,a=270.,b=270.,width=20.,cut=1.)
ring4=GaussianRing(amplitude=0.0005,a=40.,b=40.,width=5.,cut=1.)
centralgauss=models.Gaussian2D(amplitude=0.0035, x_mean=0., y_mean=0., x_stddev=5., y_stddev=5., theta=0.)



ring1.theta.fixed=True
ring2.theta.fixed=True
ring3.theta.fixed=True
ring4.theta.fixed=True
centralgauss.theta.fixed=True

model=centralgauss+ring1+ring2+ring3+ring4

modelend=fit_LSQ(model,xx,yy,image)
#showi(modelend(xx,yy))

modelend.theta_0.fixed=False
modelend.theta_2.fixed=False
modelend.theta_1.fixed=False
modelend.theta_3.fixed=False
modelend.theta_4.fixed=False


modelend=fit_LSQ(modelend,xx,yy,image)
modelend=fit_LSQ(modelend,xx,yy,image)
#showi(modelend(xx,yy))




#modelplot2(modelend(xx,yy),image,'inferno',header)


thetamax=np.array([ 0.005, 5, 5,  20,
        20, 2*np.pi,  0.01,  0.01,
        2, 2, 2, 10,
        10, 100, 100, np.pi,
        0.001, 10, 10, 50,
        300, 300,  np.pi,  1.01,
        0.001, 10,10, 200,
        300, 300, np.pi, 1.01])

thetamin=np.array([ 0,-5,-5, 5,
        5,-np.pi,0,0,
        0.01,0.01,0.01,-10,
        -10,10,10,-np.pi,
        0,-10,-10,10,
        200,200,-np.pi,0.01,
        0,-10,-10,10,
        200,200,-np.pi,0.01])

def lnprior(theta):
    if np.all((thetamin<theta)*(theta<thetamax)):
        return(0.0)
    return(-np.inf)

def lnlike(theta):
    modelend.parameters=theta
    model=modelend(xx,yy)
    error = 2.84e-05
    chi2=np.sum((image-model)**2)/(-2*error**2)
    return(chi2)

def lnprob(theta):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return(-np.inf)
    return(lp + lnlike(theta))

import emcee

"""
hehe=[]
for wawa in range(8,17):
    haha=[]
    for thr in range(1,9):
        ndim, nwalkers, iterations = 32, wawa*8, 2
        pos = [(1 + 1e-3*np.random.randn(ndim))*theta for i in range(nwalkers)]
        huhu = time()
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,threads=thr)
        t=sampler.run_mcmc(pos, iterations)
        haha.append(time()-huhu)
    hehe.append(haha)



fig, axes = plt.subplots(5, figsize=(10, 20), sharex=True)
samples = sampler.chain
labels = [i for i in range(5)]

for i in range(5):
    ax = axes[i]
    ax.plot(np.transpose(samples[:, :, i]), "k", alpha=0.3)
    ax.set_xlim(0, iterations)
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

"""

from emcee.utils import MPIPool
huhu=time()
pool = MPIPool()#loadbalance=True)
if not pool.is_master():
    pool.wait()
    sys.exit(0)

ndim, nwalkers, iterations = 32, 64, 2
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool)
pool.close()
print(time()-huhu)




