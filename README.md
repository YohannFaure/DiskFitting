# DiskFitting : fitting a Protoplanetary Disk

This package was designed to fit J1615 protoplanetary disk in the image plane, but is quite general and can be adapted quite easily to other disks.

## How to install the packages?
requirements :
1. Python3.7
2. Several modules : numpy, matplotlib, astropy, scipy

First, you should import my repository :
`git clone https://github.com/YohannFaure/DiskFitting.git`


Then you should install the latest version of Emcee (which might not be in the usual repos).
If you already have emcee installed with conda, you should uninstall it: `conda uninstall emcee`.


```
git clone https://github.com/dfm/emcee.git
cd emcee
python3 setup.py install
```
If you are installing system-wise, you should add `sudo` at the beggining of the last line. If you are not a sudoer, you should activate your conda environment before installing (if you don't know conda, see [here](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/)).

#### Note:
You should add a `PAR` keyword to your fits file, containing the paralax of the disk, as output by the [Gaia database](https://gea.esac.esa.int/archive/). If you don't have this piece of information, edit the code accordingly or just put a `PAR` keyword with a value of 1000. Your plots will then have axis in arcseconds instead of A.U.

## What does each file do?

### `FunctionsModule.py`

It is a module with many usefull functions. Each function is described in the code.

### `TiltFinder`

It finds the tilt of a Quasi-Gaussian image, *i.e.* the inclination and position angle of the image, using a classical gradient descent optimization

The inc and pa will then be used to compute a mesh in the rotation plane, making the computation faster.

To use it, call `python3 TiltFinder.py J1615_edit.fits`

Some options are available, described within the code.

### `Spirals.py`

This is just a toy program to point and click elements in an image.

`python3 Spirals.py /home/yohann/Desktop/Stage2019/DiskFitting/spiral.fits --inc 10 --pa 10 --center '(511.74577572964677, 512.5657729367406)'`


### `RadialProfile.py`

This computes a nice intensity radial profile, simply by calling
`python3 RadialProfile.py file inc pa --args`

for example:
`python3 RadialProfile.py J1615_edit.fits 4.59835388e+01 32.62477693e+01 12.5 --CenterAndWidth '(1500,1500,200)'`


### `AngularProfile.py`

This file defines a function used to plot angular profiles of the image. It can even be used to produce videos.

As it is quite customized, you should read and understand it before using it.

### `OptimizationModule.py`
This file contains many functions used to compute the model of J1615, especially it opens the image and initiate the Mesh to optimize in the prefered plane (inc, pa).


### `ModelingEmcee.py`

It calls `OptimizationModule.py` and starts the EMCEE optimization. You can adapt it to add more classical optimization steps.

`python3 ModelingEmcee.py nwalkers nsteps nthreads --suffix 'suffix' --resume file/to/resume`

### `...AltMethod.py`

Does the same but using a different method, based on not tilting the Mesh and using ellipses.

### `MergeOpti.py`

Can be used to merge optimization files.

### `Legacy`

Random bits of code that migh be usefull one day.

## Contact and conditions

If you need help feel free to contact me
faure(dot)yohann(at)gmail(dot)com

You can branch and improve this code freely.
