##Get distribution of peaks to identify driven from noise

from numpy import genfromtxt
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


def peakheight(data):	#takes column of y values
	#Go through y-values and identify local maxima, calculate height based on most recent minimum
	#return array of [peak x-value, peak height]
	peaks = []
	
	for x in range(1,len(data)-1):
		#check if peak
		if( data[x]>data[x-1]) and (data[x]>data[x+1]):
			i=1
			#find previous local min, x-i part keeps index in bounds
			while(data[x-i] > data[x-(i+1)]) and ((x-i)>=0):
				i+=1
		#	print(str(x) + " is a peak of height "+str(data[x]-data[x-i]))
			peaks.append([x, data[x]-data[x-i]])
	return peaks




	
data = genfromtxt('huffbuzz12.csv',delimiter=',')
data = data[1:][:,1:3] #remove top row of text,first column

xpeaks = np.array(peakheight(data[:,0]))
ypeaks = np.array(peakheight(data[:,1]))

common_params = dict(bins=100, range=(0, 100),normed=False)

common_params['histtype'] = 'step'
plt.title('Peak Size Distribution')
# the histogram of the data
n, bins, patches = plt.hist(xpeaks[:,1], facecolor='g', alpha=0.75, color='c', **common_params)

n, bins, patches = plt.hist(ypeaks[:,1], facecolor='c', alpha=0.75,color='g', **common_params)



plt.xlabel('Peak heights')
plt.ylabel('Number of Peaks at Specific Height')
#plt.axis([0, 100, 0, 10])
plt.grid(True)
plt.autoscale(enable=True)
plt.show()
