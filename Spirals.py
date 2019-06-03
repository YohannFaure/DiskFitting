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



# Simple mouse click function to store coordinates
def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    # print 'x = %d, y = %d'%(
    #     ix, iy)
    # assign global variable to access outside of function
    global coords
    coords.append((ix, iy))
    # Disconnect after 2 clicks
    if len(coords) == -1:
        fig.canvas.mpl_disconnect(cid)
        plt.close(1)
    return(None)


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description = 'arguments')
    parser.add_argument("location", help = "File to use as image data.",type = str)
    parser.add_argument("--inc", help = "Inclination of the disk, add 360 if it's negative ('-' is considered as a system prefix). (deg)",type = float)
    parser.add_argument("--pa", help = "Position angle of the disk, add 360 if it's negative ('-' is considered as a system prefix). (deg)",type = float)
    args = parser.parse_args()
    location=args.location
    header,image=openimage(location)
    coords = []
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.imshow(image)
    # Call click func
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show(1)
    center=(1500,1500)
    aaa=TiltFinder(location,center=center,width=600)
    inc,pa=aaa[2:4]
    coords=np.array(coords)
    coords=coords-np.array(center)
    x,y,r,theta=deprojectedcoordinatespairs(np.transpose(coords),inc,pa)
    plt.scatter(theta,r)
    plt.scatter(theta+2*np.pi,r)
    plt.show()

