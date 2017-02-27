from scipy.optimize import curve_fit
from scipy.integrate import odeint
import scipy as sp
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

def fn(x,e,f,g):
    return e*x[0]+f*x[1]+g*x[2]

#PART 1
#Import data and calculate fit parameters

rawdata = genfromtxt('testdata.csv',delimiter=',')
rawdata = rawdata[1:] #remove top row of text

N= len(rawdata)  
data = sp.zeros((N,4))

#Get X coefficients
#data gets X,Y and transforms it into dX/dt, X, X^2, XY
for index,x in enumerate(rawdata):
    # For dX/dt, do 2-point slope, except for first and last datapoints
    if index==0:
        data[index,0]=rawdata[index+1][0] - rawdata[index][0] 
    elif index==N-1:
        data[index,0]=rawdata[index][0] - rawdata[index-1][0]
    else:
        data[index,0]=(rawdata[index+1][0] - rawdata[index-1][0])/2
    data[index,1]=x[0] #X
    data[index,2]=x[0]**2 #X^2
    data[index,3]=x[0]*x[1] #XY

xdata = sp.array(data)
dx = sp.array(xdata[:,0])
xdata = sp.transpose(xdata[:,1:])
xpopt, xpcov = curve_fit(fn,xdata,dx) #Fit for dX/dt parameters

#Get Y Coefficients
for index,x in enumerate(rawdata):

    # For dY/dt, do 2-point slope, except for first and last datapoints
    if index==0:
        data[index,0]=rawdata[index+1][1] - rawdata[index][1] 
    elif index==N-1:
        data[index,0]=rawdata[index][1] - rawdata[index-1][1]
    else:
        data[index,0]=(rawdata[index+1][1] - rawdata[index-1][1])/2
    data[index,1]=x[1] #Y
    data[index,2]=x[1]**2 #Y^2
    data[index,3]=x[0]*x[1] #XY

#Put ydata into scipy array
ydata = sp.array(data)
dy = sp.array(ydata[:,0]) #Get dY/dt values from first column
ydata = sp.transpose(ydata[:,1:]) #Transpose ydata for curve_fit

ypopt, ypcov = curve_fit(fn,ydata,dy)#Fit parameters for Y,Y^2,XY to dY/dT

#turn values into recognizable coefficients in the form [r1, k1, a1]
xpopt[1]= -1*xpopt[0]/xpopt[1]
ypopt[1]= -1*ypopt[0]/ypopt[1]

print(xpopt)
print(ypopt)

#standard deviations
stdx = np.sqrt(np.diag(xpcov))
stdy = np.sqrt(np.diag(ypcov)) 

#define error as sqrt((Rstd/R)^2+(Mstd/M)^2+...)
error= sum(np.square(np.divide(np.concatenate([stdx,stdy]),np.concatenate([xpopt,ypopt]))))**(.5)
print("Error: "+str(error))

r = [xpopt[0],ypopt[0]]
M = [xpopt[1],ypopt[1]]
a = [-1*xpopt[2],-1*ypopt[2]]



#PART 2
#Plug coefficients into dif-eq and solve it

def species(y,t):   #f0, f1 = dX/dt, dY/dt;   r,M,a are [rx,ry], [Mx,My], etc
    f0 = r[0]*y[0]*(1-y[0]/M[0])-a[0]*y[0]*y[1]
    f1 = r[1]*y[1]*(1-y[1]/M[1])-a[1]*y[0]*y[1]
    return [f0,f1]

#Get IC from original data
X0=rawdata[0][0]
Y0=rawdata[0][1]
t = np.linspace(0,N,10*N) #number of months to simulate

soln = odeint(species, [X0,Y0],t)#solve ODEs
fig = plt.figure()
ax1 = fig.add_subplot(111)

#Only plot dX/dt at the moment
ax1.plot(t,soln[:, 0], c='b', label='Fit X')
ax1.plot(t,soln[:, 1], c='r', label='Fit Y')

ax1.plot(range(N),rawdata[:,0], c='c', label='Data X')
ax1.plot(range(N),rawdata[:,1], c='g', label='Data Y')

#make parameters human readable
textstring = ''
for i in r+M+a:
    textstring+="{0:.5f}".format(i)+", "

ax1.set_title('r1, r2, M1, M2 ,a1, a2\n'+textstring)
legend = ax1.legend(loc='upper center', shadow=False)

plt.show()
