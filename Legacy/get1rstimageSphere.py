from TiltFinder import *
import matplotlib as mpl
import matplotlib.pyplot as plt
#9.96829

location="J1615_sphere.fits"
header,image=openimage(location)
location2='J1615_edit.fits'
header2,image2=openimage(location2)
pixtosec=12.25e-3/2
par=header2['PAR']
pixtoau=1000*pixtosec/(par)
arcsectoau=1000/par
#image=resizedimage(image,1500,1500,400)
scalex=pixtoau*250
scaley=pixtoau*250

mmm=np.max(image)
image=image/mmm*10

import matplotlib as mpl
mpl.use('pgf')
pgf_with_rc_fonts = {
    "font.family": "serif",
    "font.serif": [],                   # use latex default serif font
    "font.sans-serif": [], # use a specific sans-serif font
    "font.size": 8}

mpl.rcParams.update(pgf_with_rc_fonts)
plt.style.use('dark_background')



fig,ax=plt.subplots()
fig.set_size_inches(2.98,2.7)

im=ax.imshow(image, interpolation='nearest',origin=0,cmap='inferno',extent=[scalex,-scalex,-scaley, scaley],clim=(0,7e-6/mmm*10))
#beam = mpl.patches.Ellipse(xy=(scalex*0.9,-scaley*0.9), width=header['BMIN']*3600*arcsectoau, height=header['BMAJ']*3600*arcsectoau , color='white', fill=True, angle=header['BPA'])
#ax.add_artist(beam)
ax.set_xlabel('East-West (au)')
ax.set_ylabel('South-North (au)')
from mpl_toolkits.axes_grid1 import make_axes_locatable


divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)
plt.tight_layout()
plt.savefig('SimpleImageSphere.pgf')
