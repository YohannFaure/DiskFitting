from OptimizationModule import *
from multiprocessing import Pool

import os
os.environ["OMP_NUM_THREADS"] = "1"


ring1=FFRing(i0=0.0036,i1=0.0036,sig0=1.,sig1=0.1,gam=1.6,xc=0.,yc=0.,a=50.,b=50.,theta=0.)
ring2=GaussianRing(amplitude=0.0005,a=230.,b=230.,width=20.)
ring3=GaussianRing(amplitude=0.0004,a=270.,b=270.,width=20.)
ring4=GaussianRing(amplitude=0.0005,a=40.,b=40.,width=5.)
centralgauss=Gaussian2D(amplitude=0.0035, x_mean=0., y_mean=0., x_stddev=5., y_stddev=5., theta=0.)
model=centralgauss+ring1+ring2+ring3+ring4



theta=np.array([ 2.23181626e-03, -1.07790689e+00, -4.16901180e+00,  1.59107549e+01,
        1.13123755e+01, -1.02020398e-01,                                         #Gaussian2D
        3.58830488e-03,  1.42910367e-03,  1.04998622e+00,  3.93519539e-01,
        1.94025614e+00,  5.16452184e-01,  3.43353710e-01,  5.30671757e+01,
        5.39745428e+01, -4.75691404e-02,                                         #FFRing
        1.74619189e-04, -3.80317291e+00, -1.36541282e+00,  2.29360533e+01,
        2.20108561e+02,  2.27487826e+02, -2.05849846e-01,                        #GaussianRing
        5.12031052e-04,  2.64237935e+00, -1.66654845e+00,  8.76122538e+01,
        2.32509969e+02,  2.38602659e+02,  1.82304211e-02,                        #GaussianRing
        1.52748894e-03, -1.66277410e+00, -1.19165019e+00,  1.48840029e+01,
        4.82086253e+01,  5.09698076e+01,  1.22922435e-01])                       #GaussianRing

"""

theta=ClassicalOptimization(model)

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


model.parameters = theta

#extract=[0,3,4,6,7,8,9,10,13,14,16,19,20,21,24,27,28,29]
extract=list(range(len(theta)))
#extract=[1,2]
thetabis=theta[extract]
thetaminbis=thetamin[extract]
thetamaxbis=thetamax[extract]

ndim, nwalkers, iterations, nthread = len(extract), 100,200, 8
error = (2.84e-05)
sigerror=1/(-2.*error**2.)
def lnprior(thetabis):
    if np.all((thetaminbis<thetabis)*(thetabis<thetamaxbis)):
        return((0.))
    return((-np.inf))

def lnlike(thetabis):
    model.parameters[extract]=thetabis
    modeling=model(xx,yy)
    return(sigerror*np.sum((image-modeling)**2.))


def lnprob(thetabis):
    lp = lnprior(thetabis)
    if not np.isfinite(lp):
        return((-np.inf))
    return(lp + lnlike(thetabis))

pos = np.array([(1. + 1e-3*np.random.randn(ndim))*thetabis for i in range(nwalkers)])


with Pool(processes=nthread) as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool)
    t=sampler.run_mcmc(pos, iterations, progress=True)

samples = sampler.chain

np.save("results/optimization/opti_{}_{}_{}.npy".format(ndim, nwalkers, iterations),(samples,thetamin,thetamax,model.param_names))

