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

@models.custom_model
def GaussianRing(xx, yy, amplitude=1., xc=0., yc=0., width=1., a=1.,b=1.,theta=0.):
    """
    This makes a gaussian ring centered on (xc,yc), elliptic with semi-axis a and b, and rotation theta.
    """
    c=math.cos(theta)
    s=math.sin(theta)
    # centering the mesh
    x=xx-xc
    y=yy-yc
    # compute distance to the ellipse
    u=np.sqrt(((x*c+y*s)/a)**2.+((y*c-x*s)/b)**2.) - 1.
    # compute gaussian
    return( amplitude * np.exp(  ( -.5*(a*b)*(width)**-2. ) * (u**2.) ) ) 


'''
@models.custom_model
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
'''

@models.custom_model
def FFRing(xx,yy,i0=1.,i1=1.,sig0=1.,sig1=1.,gam=1.,xc=0.,yc=0.,a=1.,b=1.,theta=0.):
    """
    Facchini Ring, refer to eq 1 in arXiv 1905.09204.
    """
    c=math.cos(theta)
    s=math.sin(theta)
    x=xx-xc
    y=yy-yc
    u=np.sqrt(((x*c+y*s)/a)**2.+((y*c-x*s)/b)**2.)
    f = (i0 * ((u / sig0)**gam)) * np.exp(-(u**2.) / (2. * sig0**2.))
    f += i1 * np.exp(-(u**2.) / (2. * sig1**2.))
    return(f)


@models.custom_model
def Gaussian2D(xx, yy, amplitude=1., x_mean=0., y_mean=0., x_stddev=1.,y_stddev=1.,theta=0.):
    """
    Redifining the gaussian2D Model of astropy module, for it to be faster in terms of computation. Please refer to the documentation of astropy.modeling.models.Gaussian2D
    """
    c=math.cos(theta)
    s=math.sin(theta)
    x=xx-x_mean
    y=yy-y_mean
    c2=math.cos(2.*theta)
    s2=math.sin(2.*theta)
    csq=c**2.
    ssq=s**2.
    a=.5*( csq * (x_stddev )**-2. + ssq * ( y_stddev )**-2.)
    b=.5*s2*(y_stddev**-2.-x_stddev**-2.)
    c=.5*(ssq*(x_stddev**-2.) + csq*(y_stddev**-2.))
    return(amplitude*np.exp( - (a*x**2.+b*(x*y)+c*y**2.) ))



def ClassicalOptimization(model):
    """
    Makes the classic optimization using scipy.optimize.
    """
    ##### Define the fitter
    fit_LSQ=fitting.LevMarLSQFitter()
    ##### At first fix theta so it doesn't go round https://youtu.be/PGNiXGX2nLU?t=60
    model.theta_0.fixed=True
    model.theta_2.fixed=True
    model.theta_1.fixed=True
    model.theta_3.fixed=True
    model.theta_4.fixed=True
    ##### Optimize
    model=fit_LSQ(model,xx,yy,image)
    print("Classical Optimization 1/3 done")
    ##### Unfix
    model.theta_0.fixed=False
    model.theta_2.fixed=False
    model.theta_1.fixed=False
    model.theta_3.fixed=False
    model.theta_4.fixed=False
    ##### Double optimization
    model=fit_LSQ(model,xx,yy,image)
    print("Classical Optimization 2/3 done")
    model=fit_LSQ(model,xx,yy,image)
    print("Classical Optimization done")
    return(model.parameters)


def EvolutionPlot(samples,nrows,ncols,figsize=(20,20),labels=None,limits=None):
    fig, axes = plt.subplots(nrows=nrows,ncols=ncols, figsize=figsize, sharex=True)
    iterations,nwalkers,ndims=samples.shape
    if labels==None:
        labels=range(ndims)
    if limits:
        thetamin,thetamax=limits
        for i in range(ndims):
            ax = axes.flatten()[i]
            _=ax.plot(np.transpose(samples[:, :, i]), "k", alpha=0.3)
            _=ax.set_xlim(0, iterations)
            _=ax.set_ylabel(labels[i])
            _=ax.yaxis.set_label_coords(-0.1, 0.5)
            _=ax.plot([0,iterations],[thetaminbis[i],thetaminbis[i]])
            _=ax.plot([0,iterations],[thetamaxbis[i],thetamaxbis[i]])
        plt.show()
    else:
        for i in range(ndims):
            ax = axes.flatten()[i]
            _=ax.plot(np.transpose(samples[:, :, i]), "k", alpha=0.3)
            _=ax.set_xlim(0, iterations)
            _=ax.set_ylabel(labels[i])
            _=ax.yaxis.set_label_coords(-0.1, 0.5)
        plt.show()
    return(None)


def CornerPlot(samples):
    import corner
    figure = corner.corner(samples)
    
