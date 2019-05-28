from FunctionsModule import *
header,image=openimage('/home/yohann/Desktop/Stage2019/code/J1615_LB_ap_0.5.fits')
inc,pa=4.59835388e+01,-3.42477693e+01
degtorad=np.pi/180.
radtodeg=180./np.pi
lx,ly=image.shape
image=resizedimage(image,lx//2,ly//2,500)
llx,lly=image.shape
x = np.arange(llx)-llx/2
y = np.arange(lly)-llx/2
x, y = np.meshgrid(x, y)

pixtosec=np.abs(3600*header['CDELT1'])

xx,yy=deproject(x,y,inc*degtorad,pa*degtorad)
rp=np.sqrt(xx**2+yy**2)

#from mpl_toolkits.mplot3d import Axes3D
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.plot_surface(xx, yy, image,cmap='inferno',interpolation=nearest)
#plt.show()

fig = plt.figure()
fig.set_size_inches(5,5)
ax = fig.add_subplot(1, 1, 1)
ax.set_facecolor('grey')
fig.suptitle('Our protoplanetary disk front-viewed',size=12)
ax.contourf(xx*pixtosec,yy*pixtosec,image,cmap='inferno',levels=1000,interpolation='nearest')
ax.axis('equal')
ax.set_xlabel(r"$\vec{u_x}$ (arcsec)")
ax.set_ylabel(r"$\vec{u_y}$ (arcsec)")
plt.xlim(np.max(xx)*pixtosec,np.min(xx)*pixtosec)
#plt.tight_layout()

beam = mpl.patches.Ellipse(xy=(2.5,-2.5), width=header['BMIN']*3600, height=header['BMAJ']*3600 , color='white', fill=True, angle=header['BPA'])
ax.add_artist(beam)

plt.savefig('frontview.png',dpi=600)
