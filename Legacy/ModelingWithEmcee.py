from FunctionsModule import *
from astropy.modeling import models, fitting
from astropy.convolution import convolve_models
import emcee
from emcee.utils import MPIPool


location="J1615_edit.fits"

xc,yc,inc,pa=296.7947187252183, 298.211358761549, 0.7954129826814671, 2.5385352672838146
header,image=openimage(location)
image=resizedimage(image,1500,1500,400)
x,y=meshmaker(800,800,xc+100,yc+100)
xx,yy,rr,angles=deprojectedcoordinates(x,y,inc,pa)
pixtosec,pixtoau,arcsectoau=GlobalVars(header)

##### Define the beam convolution
beamgauss = models.Gaussian2D(amplitude=1, x_mean=0., y_mean=0., x_stddev=header['BMIN']*3600/pixtosec, y_stddev=header['BMAJ']*3600/pixtosec, theta=header['BPA']*degtorad)



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

ring1=FFRing()
ring2=GaussianRing()
ring3=GaussianRing()
centralgauss=models.Gaussian2D()
model=centralgauss+ring1+ring2+ring3



theta=np.array([ 2.59465343e-03, -4.04282496e-01, -2.72041882e+00,  1.17685187e+01,
        9.09356132e+00,  1.48009232e-01,  5.30247428e-03,  6.25817419e-04,
        1.11292009e+00,  1.24025793e-01,  8.28538127e-01,  4.25793371e-03,
        4.19708902e-01,  5.69014124e+01,  5.82810972e+01,  1.62138909e-02,
        2.77134498e-04,  9.26023617e-01, -1.34757436e+00,  4.25347893e+01,
        2.31211713e+02,  2.42679913e+02,  5.13938504e-02,  1.00000000e+00,
        3.61491361e-04,  1.74556442e+00, -9.95154665e-01,  1.06450020e+02,
        2.30275755e+02,  2.33531565e+02, -1.01574007e-01,  1.00000000e+00])

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

model.parameters = theta

def lnprior(theta):
    if np.all((thetamin<theta)*(theta<thetamax)):
        return(0.0)
    return(-np.inf)

def lnlike(theta):
    model.parameters=theta
    modeling=model(xx,yy)
    error = 2.84e-05
    chi2=np.sum((image-modeling)**2)/(-2*error**2)
    return(chi2)

def lnprob(theta):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return(-np.inf)
    return(lp + lnlike(theta))


from emcee.utils import MPIPool

huhu=time()
pool = MPIPool()#loadbalance=True)
if not pool.is_master():
    pool.wait()
    sys.exit(0)

ndim, nwalkers, iterations = 32, 256, 10
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool)
pos = [(1 + 1e-3*np.random.randn(ndim))*theta for i in range(nwalkers)]
t=sampler.run_mcmc(pos, iterations)
print(time()-huhu)
print(sampler.chain.shape)
pool.close()
print(time()-huhu)
print(sampler.chain.shape)

