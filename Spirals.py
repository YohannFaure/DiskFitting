#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Just an example on how to point-click a spiral

python3 Spirals.py /home/yohann/Desktop/Stage2019/DiskFitting/spiral.fits --inc 10 --pa 10 --center '(511.74577572964677, 512.5657729367406)'

python3 Spirals.py /home/yohann/Desktop/Stage2019/DiskFitting/spiral.fits --center '(511.74577572964677, 512.5657729367406)'

python3 Spirals.py /home/yohann/Desktop/Stage2019/DiskFitting/J1615_edit.fits --TiltFinderWidth 600



The points you click will end up in the coords variable.

To select a point, put your mouse on it and press space.

"""

import numpy as np
import matplotlib.pyplot as plt
from FunctionsModule import *
from TiltFinder import *

##### The points will be savec here
coords = []

ploteddots=[]
# Simple mouse click function to store coordinates
def onrightclick(event):
    if event.button==3:
        print('deleted ',coords.pop())
        t=ploteddots.pop().remove()
        fig.canvas.draw()
    return(None)

def onkeypress(event):
    ix, iy = event.xdata, event.ydata
    coords.append((ix, iy))
    print('added   ',coords[-1])
    ploteddots.append(plt.plot(ix,iy, 'x')[0])
    fig.canvas.draw()
    return(None)


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description = 'arguments')
    parser.add_argument("location", help = "File to use as image data.",type = str)
    parser.add_argument("--TiltFinderWidth", help = "finds inc and pa if the disk is gaussian enough, with the given width. Makes --inc and --pa useless.",type=int,default=None)
    parser.add_argument("--inc", help = "Inclination of the disk, add 360 if it's negative ('-' is considered as a system prefix). (deg)",type = float,default=0.)
    parser.add_argument("--pa", help = "Position angle of the disk, add 360 if it's negative ('-' is considered as a system prefix). (deg)",type = float,default=0.)
    parser.add_argument("--center", help = "Center of the disk. (pixel)",type = str,default=None)
    args = parser.parse_args()
    ##### open image
    location=args.location
    header,image=openimage(location)
    ##### create the plot
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.set_title('To add a point, put your cursor on it\nand press the space bar.\nTo remove a point, right click.')
    ax.imshow(image)
    # Call click function
    cid = fig.canvas.mpl_connect('key_press_event',onkeypress)
    cid2= fig.canvas.mpl_connect('button_press_event',onrightclick)
    plt.show(1)
    ##### compute the deprojected values
    if args.center:
        center=eval(args.center)
    else:
        center = tuple((np.array(image.shape)/2).astype(int))
        print(center)
    if args.TiltFinderWidth:
        ##### If no inc and pa given, 
        width=args.TiltFinderWidth
        aaa=TiltFinder(location,center=center,width=width)
        inc,pa=aaa[2:4]
        center = (center[0]+aaa[0]-width/2,center[1]+aaa[1]-width/2)
    else :
        inc,pa=args.inc*degtorad,args.pa*degtorad
    ##### Plot the coords
    coords=np.array(coords)
    coords=coords-np.array(center)
    x,y,r,theta=deprojectedcoordinatespairs(np.transpose(coords),inc,pa)
    plt.scatter(theta,r,marker='x')
    plt.scatter(theta+2*np.pi,r,marker='x')
    plt.show()
