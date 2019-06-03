#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Just an example on how to point-click a spiral

python3 Spirals.py /home/yohann/Desktop/Stage2019/DiskFitting/spiral.fits --inc 0 --pa 0

"""

import numpy as np
import matplotlib.pyplot as plt
from FunctionsModule import *
from TiltFinder import *

coords = [(0,0),(0,0)]


# Simple mouse click function to store coordinates
def onclick(event):
    ix, iy = event.xdata, event.ydata
    coords.append((ix, iy))
    coords.append((ix, iy))
    return(None)

def onresize(event):
    print(coords.pop())
    return(None)

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description = 'arguments')
    parser.add_argument("location", help = "File to use as image data.",type = str)
    parser.add_argument("--inc", help = "Inclination of the disk, add 360 if it's negative ('-' is considered as a system prefix). (deg)",type = float,default=None)
    parser.add_argument("--pa", help = "Position angle of the disk, add 360 if it's negative ('-' is considered as a system prefix). (deg)",type = float,default=None)
    args = parser.parse_args()
    location=args.location
    header,image=openimage(location)
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.imshow(image)
    # Call click func
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
#    cid = fig.canvas.mpl_connect('resize_event', onresize)
    cif = fig.canvas.mpl_connect('draw_event', onresize)
    plt.show(1)
    #####
    print(coords)
    center=(511.74577572964677, 512.5657729367406)
    print(args.pa,args.inc)
    if args.pa==None or args.inc==None:
        aaa=TiltFinder(location,center=center,width=600)
        inc,pa=aaa[2:4]
    else :
        inc,pa=args.inc,args.pa
    coords=np.array(coords)
    coords=coords-np.array(center)
    x,y,r,theta=deprojectedcoordinatespairs(np.transpose(coords),inc,pa)
    plt.scatter(theta,r)
    plt.scatter(theta+2*np.pi,r)
    plt.show()
