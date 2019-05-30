#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file is used to make an angular profile of an image, knowing it's inclination and position angle

r+-deltar=radius of the profile (pixels)

w = width of the image (must be higher than projected radius of the profile), the smaller the faster the program.

meann = averaging parameter, the higher the smoother
"""

from FunctionsModule import *

def AngularProfilePlot(location,inc,pa,r,deltar,w,meann):
    """It just wraps up all the process of plotting an angular profile"""
    ##### Getting info
    meann=int(meann)*2+1
    header,image=openimage(location)
    par=header["PAR"]
    pixtosec=np.abs(3600*header['CDELT1'])
    pixtoau=1000*pixtosec/(par)
    arcsectoau=1000/par

    ##### Resizing and meshing
    lx,ly=image.shape
    image=resizedimage(image,lx//2,ly//2,w)
    llx,lly=image.shape
    x, y = meshmaker(llx,lly,llx/2,lly/2)
    xx,yy,rr,angles = deprojectedcoordinates(x,y,inc*degtorad,pa*degtorad)
    ##### Compute
    coords,image=angularprofile(image,rr,angles,r,deltar)
    for i in range(len(image)):
        for j in range(len(image)):
            if np.abs(angles[i,j])<0.01:
                image[i,j]=None
            if np.abs(angles[i,j]-np.pi/2)<0.005 and rr[i,j]>200 and rr[i,j]<250:
                image[i,j]=None
    ang,val=coords.transpose()
    ang=(180/np.pi)*ang
    ##### Because it's easier
    ang,val=zip(*sorted(zip(ang, val)))

    ##### Normalization
    val=val/np.max(val)

    ##### Smoothing
    angb,valb=polynomean(ang,meann,3),polynomean(val,meann,3)

    ##### Plotting
    fig=plt.figure()
    fig.set_size_inches(10,5)

    ax1=fig.add_subplot(121)
    plt.plot(angb,valb,c='r',label='Polynomial mean')
    plt.scatter(ang,val,s=0.2,label='Scattered values')
    plt.legend()
    plt.ylim(0.,1.1)
    plt.grid()
    plt.title('Normalized azimuth intensity profile\n r={:.2f}$\pm${:.4f} (au)'.format(r*pixtoau,deltar*pixtoau),size=11)
    ax1.set_xlabel('Angle (deg)')

    ax2=fig.add_subplot(122)
    scale=pixtoau*llx/2
    plt.imshow(image,aspect='equal',vmin=0, interpolation='nearest',origin=0,cmap='inferno',extent=[scale,-scale,-scale,scale])
    plt.colorbar()
    plt.title('Profile position on the disk',size=11)
    ax2.set_xlabel('East-West (au)\nThe white ellipse is the tested zone.\n The large line is 0°.\nThe small line is 90°.')
    ax2.set_ylabel('North-south (au)')
    beam = mpl.patches.Ellipse(xy=(0.9*scale,-0.9*scale), width=header['BMIN']*3600*arcsectoau, height=header['BMAJ']*3600*arcsectoau , color='white', fill=True, angle=header['BPA'])
    ax2.add_artist(beam)
    
    ##### failed attempt of a second axis for arcseconds
    #ax3 = ax2.twiny()
    #ax3.set_xlim(scale/arcsectoau,-scale/arcsectoau)
    #ax3.set_xlabel('(arcsec)')
    #ax3.set_ylim(-scale/arcsectoau,scale/arcsectoau)
    #ax3.set_ylabel('(arcsec)')
    ra=str(r)
    while len(ra)<3:
        ra='0'+ra
    plt.savefig('prof/AngularProfileR{}.png'.format(ra),dpi=600)

if __name__=='__main__':
    ##### Inputs
    location = '/home/yohann/Desktop/Stage2019/DiskFitting/J1615_edit.fits'
    inc,pa=4.59835388e+01,-3.42477693e+01
    r=20
    deltar=5
    w=300
    meann=10
    for i in range(0,50):
        AngularProfilePlot(location,inc,pa,i,deltar,w,meann)
        plt.close()
#    AngularProfilePlot(location,inc,pa,r,deltar,w,meann)
#    plt.close()
