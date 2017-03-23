##Get distribution of peaks to identify driven from noise

from numpy import genfromtxt
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
			print(str(x) + " is a peak of height "+str(data[x]-data[x-i]))
			peaks.append([x, data[x]-data[x-i]])
	return peaks




	
data = genfromtxt('huff.csv',delimiter=',')
data = data[1:] #remove top row of text

print(peakheight(data[:,0]))			
print(peakheight(data[:,1]))




