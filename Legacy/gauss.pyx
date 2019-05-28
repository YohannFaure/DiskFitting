cimport numpy as np
import numpy as np


def CGaussianRing(x, y, amplitude=1., xc=0., yc=0., width=1., a=1.,b=1.,theta=0.):
    """
    This makes a gaussian ring centered on (xc,yc), elliptic with semi-axis a and b, and rotation theta.
    """
    c=np.cos(theta)
    s=np.sin(theta)
    # centering the mesh
    x-=xc
    y-=yc
    # compute distance to the ellipse
    u=np.sqrt(((x*c+y*s)/a)**2+((y*c-x*s)/b)**2) - 1
    # width normalization
    ff=np.sqrt(a*b)
    # compute gaussian
    return( amplitude * np.exp(  ( -.5*(ff/width)**2 ) * (u**2) ) ) 

def FFRing(x,y,i0=1.,i1=1.,sig0=1.,sig1=1.,gam=1.,xc=0.,yc=0.,a=1.,b=1.,theta=0.):
    """
    Facchini Ring, refer to eq 1 in arXiv 1905.09204.
    """
    c=np.cos(theta)
    s=np.sin(theta)
    x-=xc
    y-=yc
    xs,ys=x.shape
    u=np.empty((xs,ys))
    for i in range(xs):
        for j in range(ys):
            u[i,j]=np.sqrt(((x[i,j]*c+y[i,j]*s)/a)**2.+((y[i,j]*c-x[i,j]*s)/b)**2.)
    f = (i0 * ((u / sig0)**gam)) * np.exp(-(u**2.) / (2. * sig0**2.))
    f += i1 * np.exp(-(u**2.) / (2. * sig1**2.))
    return(f)
