from FunctionsModule import *


header,image=openimage('/home/yohann/Desktop/Stage2019/code/J1615_LB_ap_0.5.fits')
inc,pa=4.59835388e+01,-3.42477693e+01
lx,ly=image.shape
image=resizedimage(image,lx//2,ly//2,100)
llx,lly=image.shape
x = np.arange(llx)-llx/2
y = np.arange(lly)-llx/2
x, y = np.meshgrid(x, y)
pixtosec=np.abs(3600*header['CDELT1'])
binwidth=2
xx,yy,rr,angles=deprojectedcoordinates(x,y,inc,pa)



"""
m,s=radialbin(image,rr,binwidth)
n=int(np.max(rr)//binwidth)
r=np.array(list(range(n+1)))*binwidth

fig=plt.figure()
fig.set_size_inches(5,4)
plt.errorbar(r,m,yerr=s,barsabove=True,marker='o', linestyle='')
plt.grid(True)
plt.title("Average intensity on fixed radius",size=12)
plt.xlabel('Radius (pixel)')
plt.ylabel('Mean flux')
plt.tight_layout()
plt.savefig('radialprofileZoomZoomZoom.png',dpi=600)
"""




coords=angularprofile(image,rr,angles,200,10)



