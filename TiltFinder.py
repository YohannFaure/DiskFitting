#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
TiltFinder.py

This program fings the tilt (inclination and position angle) of
a protoplanetary disk in an image.

for more detail, use python help or contact
Yohann Faure
yohann.faure@ens-lyon.fr



Launching example :
python3 TiltFinder.py file --args
python3 TiltFinder.py J1615_edit.fits --width 1400 --save False
'''

from FunctionsModule import *



def modelplot(model,image,blur,cmap,header,headerscale,sigma,params,maximage,size=(7,6),save='Quadplot{:.0f}.png'.format(time())):
    """
    Plots a quadri-images graph, need to be adapted to your own plot, as it is higgly custom...
    """
    ##### settings
    fig,axarr=plt.subplots(2,2)
    plt.suptitle('Gaussian fitting of a tilted Gaussian-like profile\n(normalization: {:5.5f} {})'.format(maximage,header['BUNIT']),fontsize=14)
    fig.set_size_inches(size[0],size[1])
    ##### scale for pixel--arcsec conversion
    scale_alma = header['CDELT2']*3600
    aus_alma    = (header['NAXIS1']/2.) * scale_alma
    pixtosec=np.abs(3600*header['CDELT1'])
    par=header['PAR']
    pixtoau=1000*pixtosec/(par)
    arcsectoau=1000/par
    scalex=aus_alma*headerscale[0]*arcsectoau
    scaley=aus_alma*headerscale[1]*arcsectoau
    ##### setting and plotting each subplots
    ### Titles
    axarr[0,0].title.set_text(r'Gaussian fit')
    axarr[0,1].title.set_text(r'Blurred image ($\sigma$='+str(sigma)+' pixel)')
    axarr[1,0].title.set_text(r'Original image')
    axarr[1,1].title.set_text(r'Residual absolute value')
    ### Plots
    axarr[0,0].imshow(model, vmin=0, vmax=1, interpolation='nearest',origin=0,cmap=cmap,extent=[scalex,-scalex,-scaley, scaley])
    axarr[0,1].imshow(blur, vmin=0, vmax=1, interpolation='nearest',origin=0,cmap=cmap,extent=[scalex,-scalex,-scaley, scaley])
    axarr[1,0].imshow(image, vmin=0, vmax=1, interpolation='nearest',origin=0,cmap=cmap,extent=[scalex,-scalex,-scaley, scaley])
    ### resolution bean on the 3rd subplot
    beam = mpl.patches.Ellipse(xy=(1.75*arcsectoau,-1.75*arcsectoau), width=header['BMIN']*3600*arcsectoau, height=header['BMAJ']*3600*arcsectoau , color='white', fill=True, angle=header['BPA'])
    axarr[1,0].add_artist(beam)
    ### legend
    axarr[1,0].set_xlabel('East-West (au)')
    axarr[1,0].set_ylabel('South-North (au)')
    axarr[1,1].set_xlabel('Inclination: {inc:3.2f}. Position angle: {pa:3.2f}. (deg)'.format(inc=params[2]*radtodeg,pa=params[3]*radtodeg))
    ### last subplot
    im=axarr[1,1].imshow(np.absolute(image - model), vmin=0, vmax=1, interpolation='nearest',origin=0,cmap=cmap,extent=[scalex,-scalex,-scaley, scaley])
    ##### layout and colorbar
    plt.tight_layout()
    #plt.grid()
    plt.subplots_adjust(top=0.85)
    cax = plt.axes([0.93, 0.1, 0.015, 0.7])
    plt.colorbar(im,cax=cax)
    if save!='False':
        fig.savefig(save,dpi=600)
    else:
        plt.show()
    return(None)

def TiltFinder(location,cmap='inferno',center=None,width=None,sigma=20,method='Powell',noplot=False,seed=[-1,-1,-1,-1,-1,-1]):
    """
    Finds the tilt of a tilted 2D gaussian, and by extension the tilt of a protoplanetary disk even if it tends to have holles
    You can help the fit by specifying the center of the disk (center = (x,y)), reducing the size of the image (width, pixel), and specify a seed.
    See code for more details.
    """
    ##### Opening the image properly
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
        size=min(lx,ly)//2
    ##### Resizing it
    image=resizedimage(image,xc,yc,size)
    llx,lly=image.shape
    headerscale=(llx/lx,lly/ly) #keep the scale for plotting
    maximage=np.max(image)
    image=image/np.max(image) #normalize image, to get better results with the minimization
    ##### Getting user seed for the minimization
    if seed[0] == -1:
        rx,ry=llx//2,lly//2
        seed = np.array([rx,ry,0,0,1.,rx//5])
    ##### Defining the mesh
    mesh=meshmaker(llx,lly,0,0)
    ##### Defining the cost and minimization
    normalizer = np.array([llx,lly,1.,1.,1.,llx/5])
    ##### Blurring image and fitting
    blur = blurimage(image,sigma = sigma)
    def TiltFinderMinimize(paramsnorm):
        print(paramsnorm)
        params=paramsnorm*normalizer
        model = gaussianfit(params, mesh)
        return(quadcost(blur,model))
    print('\n\nWorking on it\n\n')
    seednorm=seed/normalizer
    test = opt.minimize(TiltFinderMinimize,seednorm,method=method)
    params = test['x']*normalizer
#    params[2]=((params[2]+.5*np.pi)%(np.pi))-.5*np.pi
#    params[3]=((params[3]+.5*np.pi)%(np.pi))-.5*np.pi
    print('Result of the minimisation :\n\n')
    print(test)
    print('\n\n [ X center, Y center, inclination, position angle, Gaussian Amplitude, Gaussian sigma ]\n\n')
    print(params)
    print('\n')
    print(params*[pixtosec,pixtosec,radtodeg,radtodeg,maximage,pixtosec])
    print('in arcsec,arcsec,deg,deg,flux,arcsec')
    if not noplot:
        model = gaussianfit(params, mesh)
        modelplot(model,image,blur,cmap,header,headerscale,sigma,params,maximage,save='False')
    return(params)



if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description = 'arguments')
    parser.add_argument("location", help = "File to use as image data.",type = str)
    parser.add_argument("--cmap", help = "Colormap tu use for plotting.",type = str,default='inferno')
    parser.add_argument("--center", help = "Approximate center of the disk (in pixels).",type = tuple,default=None)
    parser.add_argument("--width", help = "Approximate pixel width of the zone of interest in the image.",type = int,default=None)
    parser.add_argument("--blur", help = "Level of blurring in the detection.",type = int,default=20)
    parser.add_argument("--method", help = "Method of minimization in the scipy.optimize.minimize function.",type = str,default='Powell')
    parser.add_argument("--noplot", help = "Will not plot the result and only give it if present.",action = "store_true")
    parser.add_argument("--seed",help='A research seed if you have an idea of what you are looking for',type=str,default='[-1,-1,-1,-1,-1,-1]')
    args = parser.parse_args()
    TiltFinder(args.location,cmap=args.cmap,center=args.center,width=args.width,sigma=args.blur,method=args.method,noplot=args.noplot,seed=eval(args.seed))
