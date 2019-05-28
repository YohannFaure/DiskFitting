from FunctionsModule import *
from astropy.modeling import models, fitting
import emcee
import math

##### Opening the data
location="J1615_edit.fits"
xc,yc,inc,pa=296.7947187252183, 298.211358761549, 0.7954129826814671, 2.5385352672838146
header,image=openimage(location)
image=resizedimage(image,1500,1500,400)
##### Making the mesh
x,y=meshmaker(800,800,xc+100,yc+100)
xx,yy,rr,angles=deprojectedcoordinates(x,y,inc,pa)
pixtosec,pixtoau,arcsectoau=GlobalVars(header)

##### Defining the functions

def GaussianRing(x, y, amplitude=1., xc=0., yc=0., width=1., a=1.,b=1.,theta=0.):
    """
    This makes a gaussian ring centered on (xc,yc), elliptic with semi-axis a and b, and rotation theta.
    """
    c=math.cos(theta)
    s=math.sin(theta)
    # centering the mesh
    x-=xc
    y-=yc
    # compute distance to the ellipse
    u=np.sqrt(((x*c+y*s)/a)**2.+((y*c-x*s)/b)**2.) - 1.
    # compute gaussian
    return( amplitude * np.exp(  ( -.5*(a*b)*(width)**-2. ) * (u**2.) ) ) 

def LinearRing(x, y, amplitude=1., xc=0., yc=0., a=1.,b=1.,theta=0.,end=200.):
    """
    This is vestigial. It's a ring designed to fit a linear by parts profile. It is non-physical and should therefore not be used.
    """
    # Same as GaussianRing
    c=math.cos(theta)
    s=math.sin(theta)
    xx=x-xc
    yy=y-yc
    u=np.sqrt(((xx*c+yy*s)/a)**2.+((yy*c-xx*s)/b)**2.) - 1.
    # Define the end of the second linear part
    end=end/np.sqrt(a*b)
    interbool=(u>0.)
    boolout=np.logical_and(interbool,(u<end))
    # Apply the linear mask
    ampout = boolout*(end-u)/end
    ampin = np.invert(interbool)*(-np.min(u)+u)/(-np.min(u))
    amp = ampout+ampin
    return(amp*amplitude)



def FFRing(x,y,i0=1.,i1=1.,sig0=1.,sig1=1.,gam=1.,xc=0.,yc=0.,a=1.,b=1.,theta=0.):
    """
    Facchini Ring, refer to eq 1 in arXiv 1905.09204.
    """
    c=math.cos(theta)
    s=math.sin(theta)
    x-=xc
    y-=yc
    u=np.sqrt(((x*c+y*s)/a)**2.+((y*c-x*s)/b)**2.)
    f = (i0 * ((u / sig0)**gam)) * np.exp(-(u**2.) / (2. * sig0**2.))
    f += i1 * np.exp(-(u**2.) / (2. * sig1**2.))
    return(f)



def Gaussian2D(x, y, amplitude=1., x_mean=0., y_mean=0., x_stddev=1.,y_stddev=1.,theta=0.):
    """
    Redifining the gaussian2D Model of astropy module, for it to be faster in terms of computation. Please refer to the documentation of astropy.modeling.models.Gaussian2D
    """
    c=math.cos(theta)
    s=math.sin(theta)
    x-=x_mean
    y-=y_mean
    c2=math.cos(2.*theta)
    s2=math.sin(2.*theta)
    csq=c**2.
    ssq=s**2.
    a=.5*( csq * (x_stddev )**-2. + ssq * ( y_stddev )**-2.)
    b=.5*s2*(y_stddev**-2.-x_stddev**-2.)
    c=.5*(ssq*(x_stddev**-2.) + csq*(y_stddev**-2.))
    return(amplitude*np.exp( - (a*x**2.+b*(x*y)+c*y**2.) ))



def ClassicalOptimization():
    fit_LSQ=fitting.LevMarLSQFitter()
    ring1=FFRing(i0=0.0036,i1=0.0036,sig0=1.,sig1=0.1,gam=1.6,xc=0.,yc=0.,a=50.,b=50.,theta=0.)
    ring2=GaussianRing(amplitude=0.0005,a=230.,b=230.,width=20.)
    ring3=GaussianRing(amplitude=0.0004,a=270.,b=270.,width=20.)
    ring4=GaussianRing(amplitude=0.0005,a=40.,b=40.,width=5.)
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
    return(modelend.parameters)



import os
os.environ["OMP_NUM_THREADS"] = "1"

def model(x,y,theta):
    amplitude_0, x_mean_0, y_mean_0, x_stddev_0,y_stddev_0,theta_0,i0_1,i1_1,sig0_1,sig1_1,gam_1,xc_1,yc_1,a_1,b_1,theta_1,amplitude_2, xc_2, yc_2, width_2, a_2,b_2,theta_2,amplitude_3, xc_3, yc_3, width_3, a_3,b_3,theta_3,amplitude_4, xc_4, yc_4, width_4, a_4,b_4,theta_4 = theta
    return(Gaussian2D(x,y,amplitude_0, x_mean_0, y_mean_0, x_stddev_0,y_stddev_0,theta_0)
        +FFRing(x,y,i0_1,i1_1,sig0_1,sig1_1,gam_1,xc_1,yc_1,a_1,b_1,theta_1)
        +GaussianRing(x,y,amplitude_2, xc_2, yc_2, width_2, a_2,b_2,theta_2)
        +GaussianRing(x,y,amplitude_3, xc_3, yc_3, width_3, a_3,b_3,theta_3)
        +GaussianRing(x,y,amplitude_4, xc_4, yc_4, width_4, a_4,b_4,theta_4))


theta=np.array([ 2.59465343e-03, -4.04282496e-01, -2.72041882e+00,  1.17685187e+01,
        9.09356132e+00,  1.48009232e-01, #Gaussian2D
        5.30247428e-03,  6.25817419e-04,  1.11292009e+00,  1.24025793e-01,
        8.28538127e-01,  4.25793371e-03,  4.19708902e-01,  5.69014124e+01,
        5.82810972e+01,  1.62138909e-02, #FFRing
        2.77134498e-04,  9.26023617e-01, -1.34757436e+00,  4.25347893e+01,
        2.31211713e+02,  2.42679913e+02,  5.13938504e-02, #GaussianRing
        3.61491361e-04,  1.74556442e+00, -9.95154665e-01,  1.06450020e+02,
        2.30275755e+02,  2.33531565e+02, -1.01574007e-01, #GaussianRing
        1.45872034e-03, -1.74478967e+00, -1.29192802e+00,  1.46579908e+01,
        4.81938121e+01,  5.10059507e+01,  1.26945522e-01])#GaussianRing
"""

theta=ClassicalOptimization()

"""

thetamax=np.array([ 0.005, 5, 5, 20, 20, np.pi+0.1, #
        0.01, 0.01, 2, 2, 2, 10, 10, 100, 100, np.pi+0.1, #
        0.001, 10, 10, 70, 300, 300, np.pi+0.1, #
        0.001, 10, 10, 200, 300, 300, np.pi+0.1, #
        0.01, 5, 5, 25, 100, 100, np.pi+0.01],dtype=np.float64) #

thetamin=np.array([ 0., -5 ,-5 , 5, 5, -np.pi-0.1, #
        0, 0, 0.01, 0.01, 0.01, -10, -10, 10, 10, -np.pi-0.1, #
        0, -10, -10, 10, 200, 200, -np.pi-0.1, #
        0, -10, -10, 10, 200, 200, -np.pi-0.1, #
        0.0001, -5, -5, 5, 10, 10, -np.pi-0.01],dtype=np.float64) #


#model.parameters = theta

#extract=[0,3,4,6,7,8,9,10,13,14,16,19,20,21,24,27,28,29]
extract=list(range(len(theta)))
#extract=[1,2,3]
thetabis=theta[extract]
thetaminbis=thetamin[extract]
thetamaxbis=thetamax[extract]

ndim, nwalkers, iterations, nthread = len(extract), 100,2, 8
error = (2.84e-05)
sigerror=1/(-2.*error**2.)
def lnprior(thetabis):
    if np.all((thetaminbis<thetabis)*(thetabis<thetamaxbis)):
        return((0.))
    return((-np.inf))

def lnlike(thetabis):
    return(sigerror*np.sum((image-model(xx,yy,thetabis))**2.))


def lnprob(thetabis):
    lp = lnprior(thetabis)
    if not np.isfinite(lp):
        return((-np.inf))
    return(lp + lnlike(thetabis))

pos = np.array([(1. + 1e-3*np.random.randn(ndim))*thetabis for i in range(nwalkers)])


#with Pool(processes=nthread) as pool:
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)#, pool=pool)
t=sampler.run_mcmc(pos, iterations, progress=True)

samples = sampler.chain

"""
fig, axes = plt.subplots(nrows=10,ncols=4, figsize=(20, 25), sharex=True)
labels = extract

for i in range(ndim):
    ax = axes.flatten()[i]
    _=ax.plot(np.transpose(samples[:, :, i]), "k", alpha=0.3)
    _=ax.set_xlim(0, iterations)
    _=ax.set_ylabel(labels[i])
    _=ax.yaxis.set_label_coords(-0.1, 0.5)
    _=ax.plot([0,iterations],[thetaminbis[i],thetaminbis[i]])
    _=ax.plot([0,iterations],[thetamaxbis[i],thetamaxbis[i]])

plt.show()

for i in range(ndim):
    ax = axes.flatten()[i]
    _=ax.plot(np.transpose(samples[:, :, i]), "k", alpha=0.3)
    _=ax.set_xlim(0, iterations)
    _=ax.set_ylabel(labels[i])
    _=ax.yaxis.set_label_coords(-0.1, 0.5)

plt.show()
"""

np.save("opti_{}_{}_{}.npy".format(ndim, nwalkers, iterations),samples)

