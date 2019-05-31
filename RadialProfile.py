#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This program computes a binned profile of intensity function of radius for an image. You can compute the inclination and position angle with TiltFinder.

For my image :
inc,pa=4.59835388e+01,-3.42477693e+01

python3 RadialProfile.py file inc pa --args
python3 RadialProfile.py J1615_edit.fits 4.59835388e+01 32.62477693e+01 12.5 --CenterAndWidth '(1500,1500,200)'

"""
#45.9835388
#326.2477693
from FunctionsModule import *



if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description = 'arguments')
    parser.add_argument("location", help = "File to use as image data.",type = str)
    parser.add_argument("inc", help = "Inclination of the disk, add 360 if it's negative ('-' is considered as a system prefix). (deg)",type = float)
    parser.add_argument("pa", help = "Position angle of the disk, add 360 if it's negative ('-' is considered as a system prefix). (deg)",type = float)
    parser.add_argument("binwidth", help = "Binning width in pixel.",type = float)
    parser.add_argument("--CenterAndWidth", help = "Center of the disk (in pixels) and width.",type = str,default=None)
    parser.add_argument("--name", help = "extension to the name of the file.",type = str,default=None)
    args = parser.parse_args()
    inc,pa=float(args.inc)%360.,float(args.pa)%360.
    header,image=openimage(args.location)
    par=header["PAR"]
    pixtosec=np.abs(3600*header['CDELT1'])
    arcsectoau=1000/par
    pixtoau=arcsectoau*pixtosec
    if args.CenterAndWidth :
        lx,ly,w=eval(args.CenterAndWidth)
        image=resizedimage(image,lx,ly,w//2)
    llx,lly=image.shape
    x = np.arange(llx)-llx/2
    y = np.arange(lly)-llx/2
    x, y = np.meshgrid(x, y)
    pixtosec=np.abs(3600*header['CDELT1'])
    xx,yy,rr,angles=deprojectedcoordinates(x,y,inc*degtorad,pa*degtorad)
    m,s=radialbin(image,rr,args.binwidth)
    n=int(np.max(rr)//args.binwidth)
    r=np.array(list(range(n+1)))*args.binwidth
    r=r*pixtoau
    fig=plt.figure()
    fig.set_size_inches(5,4)
    plt.errorbar(r,m,yerr=s,barsabove=True,linestyle='',markeredgewidth=2,elinewidth=0.5,c='k')
    plt.scatter(r,m,s=0.8,c='k')
    plt.grid(True)
    plt.title("Average intensity on fixed radius",size=12)
    plt.xlabel('Radius (au)')
    plt.ylabel('Mean flux ({})'.format(header['BUNIT']))
    plt.yscale('log')
    plt.tight_layout()
    plt.show()
    #plt.savefig('radialprofile{}.png'.format(args.name if (args.name) else int(time())),dpi=600)







