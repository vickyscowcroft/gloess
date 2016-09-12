#!/usr/bin/env python

import numpy as np
import numpy.ma as ma


#def fit2(x1, y1, n, wt, mwt, c1, c2, c3, sigma_c1, sigma_c2, sigma_c3, chisum, q, jstep):
def fit2(x1,y1,n,wt):
	sigma = np.zeros( (n) )
	#print sigma
	## Changing the weights passed from program to sigmas
	## and setting up some arrays
	## Can manipulate numpy arrays as a whole rather than looping

	sigma = 1.0 / wt
	sigma2 = sigma**2
	x2 = x1**2
	x3 = x1**3
	x4 = x1**4
	
	# np.nansum sums the array treating NaNs as zero
	
	C =  np.nansum(y1 / sigma2)
	E =  np.nansum((y1*x1) / sigma2)
	G = np.nansum((y1*x2) / sigma2)
	
	a11 = np.nansum(1.0 / sigma2)
	a12 = np.nansum(x1 / sigma2)
	a13 = np.nansum(x2 / sigma2)
	a23 = np.nansum(x3 / sigma2)
	a33 = np.nansum(x4 / sigma2)
	a21 = a12
	a22 = a13
	a31 = a13
	a32 = a23
	cofa11 = a22*a33-a32*a23
	cofa12 = -a21*a33+a31*a23
	cofa13 = a21*a32-a31*a22
	cofa21 = -a12*a33+a32*a13
	cofa22 = a11*a33-a31*a13
	cofa23 = -a11*a32+a31*a12
	cofa31 = a12*a23-a22*a13
	cofa32 = -a11*a23+a21*a13
	cofa33 = a11*a22-a21*a12

	det=np.abs(a11*cofa11+a12*cofa12+a13*cofa13)
	
	ai11=cofa11/det
	ai12=cofa21/det
	ai13=cofa31/det
	
	ai21=cofa12/det
	ai22=cofa22/det
	ai23=cofa32/det
	
	ai31=cofa13/det
	ai32=cofa23/det
	ai33=cofa33/det
	
	c1=ai11*C+ai12*E+ai13*G
	c2=ai21*C+ai22*E+ai23*G
	c3=ai31*C+ai32*E+ai33*G
	
	sigma_c1=np.sqrt(ai11)
	sigma_c2=np.sqrt(ai22)
	sigma_c3=np.sqrt(ai33)
	sigma_c1c2=ai21
	sigma_c1c3=ai31
	sigma_c2c3=ai23
	
	chisum = np.nansum(((c1 + c2*x1 + c3*x2 - y1)**2) / sigma2)
	
#	chisum2 = 0.0
#	for t in range(0,n):
#		print x1[n]#, x2[n], y1[n], sigma2[n]
#		chisum2 = chisum2 + ((c1 + c2*x1[n] + c3*x2[n] - y1[n])**2) / sigma2[n]
	
#	print chisum2
	return(c1)
	
def moment(data,n):
	sdev = np.std(data)
	var = np.var(data)
	ave = np.average(data)	

	diffs = data - ave
	adev = np.nansum(np.abs(diffs))
	p = diffs*diffs*diffs
	skew = np.nansum(p)
	p = p*diffs
	kurtosis = np.nansum(p)
	adev = adev / n
	if (var != 0):
		skew = skew / (n*sdev**3)
		kurtosis = kurtosis / (n*var**2) - 3.0
	amp = np.ptp(data)
	return( ave, adev, sdev, var, skew, kurtosis, amp)
	
	
def fit_one_band(data,err,phases,n,smooth):
### n parameter is redundant, phase this out
	np.seterr(divide='ignore')
	np.seterr(over='ignore')
	y = data[data<50]
	phase = phases[data<50]
	yerr = err[data<50]

	y = np.concatenate((y,y,y,y,y))
	yerr = np.concatenate((yerr,yerr,yerr,yerr,yerr))
	xphase = np.concatenate((phase,(phase+1.0),(phase+2.0),(phase+3.0),(phase+4.0)))
	size_of_data = len(y)

	dist = np.zeros( (size_of_data,500) )
	weight = np.zeros( (size_of_data) )
	x = np.zeros( (500) )
	xz = np.zeros( (size_of_data) )
	data1 = np.zeros( ( 500) )
	for count in range(0,500):
		x[count] = -0.99 + 0.01*count
		for datacount in range(0, size_of_data):
			dist[datacount,count] =  np.abs(xphase[datacount] - x[count])
			weight[datacount] = np.exp(-1.0*(dist[datacount,count]**2) /smooth**2) / yerr[datacount]
			xz[datacount] = xphase[datacount] - x[count]
		data1[count]  = fit2(xz,y,size_of_data,weight)	
	return (data1, x, y, yerr, xphase)
	
def phase_mjds(period, mjd):
	phase = (mjd / period) - np.floor(mjd / period)
	phase = np.concatenate((phase,(phase+1.0),(phase+2.0),(phase+3.0),(phase+4.0)))
	return(phase)
	
	

	
	

