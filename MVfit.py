from scipy.optimize import curve_fit
from scipy.integrate import odeint
import scipy as sp
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

def fn(x,e,f,g):
    return e*x[0]+f*x[1]+g*x[2]


rawdata = genfromtxt('testdata.csv',delimiter=',')
N = len(rawdata)-2
#data gets X,Y and transforms it into dX/dt, X, X^2, XY

data = sp.zeros((N,4))

#Get X coefficients

for index,x in enumerate(rawdata[1:-1]):
    data[index,0]=rawdata[index+2][0] - rawdata[index+1][0] #dX/dt
    data[index,1]=x[0] #X
    data[index,2]=x[0]**2 #X^2
    data[index,3]=x[0]*x[1] #XY


xdata = sp.array(data)
dx = sp.array(xdata[:,0])
xdata = sp.transpose(xdata[:,1:])

xpopt, xpcov = curve_fit(fn,xdata,dx) #Fit for dX/dt parameters

newdx = fn(xdata,xpopt[0],xpopt[1],xpopt[2])#Calculate dY/dT from new parameters

#Get Y Coefficients
#so redundant, at least it works
for index,x in enumerate(rawdata[1:-1]):
    data[index,0]=rawdata[index+2][1] - rawdata[index+1][1] #dY/dt
    data[index,1]=x[1] #Y
    data[index,2]=x[1]**2 #Y^2
    data[index,3]=x[0]*x[1] #XY

ydata = sp.array(data)
dy = sp.array(ydata[:,0])
ydata = sp.transpose(ydata[:,1:])

ypopt, ypcov = curve_fit(fn,ydata,dy)#,bounds=[(0,np.inf),(0,np.inf),(np.inf)]) #Fit for dY/dt parameters

newdy = fn(data,ypopt[0],ypopt[1],ypopt[2]) #Calculate dY/dT from new parameters

#turn values into recognizable coefficients in the form [r1, k1, a1]
xpopt[1]= -1*xpopt[0]/xpopt[1]
ypopt[1]= -1*ypopt[0]/ypopt[1]

print xpopt
print ypopt

#standard deviations
stdx = np.sqrt(np.diag(xpcov))
stdy = np.sqrt(np.diag(ypcov)) 



#define error as sqrt((Rstd/R)^2+(Mstd/M)^2+...)
error= sum(np.square(np.divide(np.concatenate([stdx,stdy]),np.concatenate([xpopt,ypopt]))))**(.5)
print "Error: "+str(error)

r = [xpopt[0],ypopt[0]]
M = [xpopt[1],ypopt[1]]
a = [-1*xpopt[2],-1*ypopt[2]]







#Now plug coefficients into dif-eq and solve it

def species(y,t):   #f0, f1 = dX/dt, dY/dt;   r,M,a are [rx,ry], [Mx,My], etc
    f0 = r[0]*y[0]*(1-y[0]/M[0])-a[0]*y[0]*y[1]
    f1 = r[1]*y[1]*(1-y[1]/M[1])-a[1]*y[0]*y[1]
    return [f0,f1]

#Get IC from original data
X0=rawdata[1][0]
Y0=rawdata[1][1]
t = np.linspace(0,N-1,100*N) #number of months to simulate

soln = odeint(species, [X0,Y0],t)#solve ODEs

fig = plt.figure()
ax1 = fig.add_subplot(111)

#Only plot dX/dt at the moment
ax1.plot(t,soln[:, 0], c='b', label='Fit X')
ax1.plot(t,soln[:, 1], c='r', label='Fit Y')
ax1.plot(range(N),rawdata[1:-1,0], c='c', label='Data X')
ax1.plot(range(N),rawdata[1:-1,1], c='g', label='Data Y')

#make parameters human readable
textstring = ''
for i in r+M+a:
    textstring+="{0:.5f}".format(i)+", "

ax1.set_title('r1, r2, M1, M2 ,a1, a2\n'+textstring)
legend = ax1.legend(loc='upper center', shadow=False)

plt.show()
