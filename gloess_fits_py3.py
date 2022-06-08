#!/usr/bin/env python

### re-write of gloess fitting functions to use linalg rather than do everything by hand. 7/6/22

## implemented auto-smoothing: finds the max distance between two consecutive phase points to decide on the smoothing parameter

## old routine still works:

## to turn off auto smoothing pass `smooth=value` to fit_one_band where `value` is a float between 0 and 1. passing nothing or `smooth="auto"` will use auto smoothing

## to use the old fitting function pass `use_matrix=False` to fit_one_band. Default is `use_matrix=True`

## Haven't done extensive testing to see whether this is quicker yet

## TO DO: 

## implement auto-shift so all lightcurves are phased with the same phi=0 at e.g. minimum of lightcurve

## working on complete re-write using classes and methods. need to make sure it's compatible with old datafiles etc. but want it to work better for stuff that's not formatted like the old gloess_in files. 




import numpy as np
import numpy.ma as ma
from scipy import linalg
from numpy.linalg import inv


def phases_to_fit(xphase):
        fake_phases = -0.99 + 0.01*(np.arange(0,500))
        fake_phases = np.reshape(fake_phases, (500, 1))
        data_phases = np.reshape(xphase, (1,len(xphase)))

        fit_phases = np.subtract(data_phases, fake_phases)
        return(fit_phases)
	
def get_weights(phase_array, yerr, smooth):
	dist  = np.abs(phase_array)
	weight = np.exp(-0.5*(dist**2) /smooth**2) / yerr 
	return(weight)
	


def fit2(x1,y1,n,wt):
	sigma = np.zeros( (n) )
	
	sigma = 1.0 / wt
	sigma2 = sigma**2
	x2 = x1**2
	x3 = x1**3
	x4 = x1**4
		
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
	
	
def fit_one_band(data,err,phases,smooth='auto', use_matrix=True):
### n parameter is redundant, phase this out
	print(f'{smooth}, {type(smooth)}, {use_matrix}')
	np.seterr(divide='ignore')
	np.seterr(over='ignore')
	## y = mags, phase = obs phases, yerr = phot errors
	y = data[data<50]
	phase = phases[data<50]
	yerr = err[data<50]

	## repeating over 5 cycles to remove boundary effects
	y = np.concatenate((y,y,y,y,y))
	yerr = np.concatenate((yerr,yerr,yerr,yerr,yerr))
	xphase = np.concatenate((phase,(phase+1.0),(phase+2.0),(phase+3.0),(phase+4.0)))
	size_of_data = len(y)

	## phase points for the fitted curve
	fake_phases = -0.99 + 0.01*(np.arange(0,500))
	## gets a matrix of distances of each observation from each phase point in the fitted curve
	phase_matrix = phases_to_fit(xphase)
	data1 = np.zeros(500)

	## auto selection of smoothing parameter if smooth='auto' (default)

	if smooth=='auto':
		sm = find_smoothing_params(xphase)
	elif (isinstance(smooth, np.floating) and smooth < 1. and smooth > 0.)  :
		sm = smooth
	else:
		print('bad smoothing parameter - using auto smoothing')
		sm = find_smoothing_params(xphase)

	for count in range(0,500):
		# phase_array = np.asarray(phase_matrix[count].flat[:])[0]
		phase_array = np.asarray(phase_matrix[count].flat[:])

		weights = get_weights(phase_array, yerr, sm)
		#w = get_weights(phase_array, yerr, sm)
		#weights = np.asarray(w[count].flat[:])[0]
		if use_matrix==True:
			#data1[count] = fit_with_matrix(phase_array, y, size_of_data, weights)
			data1[count] = fit_with_matrix(phase_array, y, weights)
		else:
			data1[count] = fit2(phase_array, y, size_of_data, weights)
	return(data1, fake_phases, y, yerr, xphase)


		
def phase_mjds(period, mjd):
	phase = (mjd / period) - np.floor(mjd / period)
	phase = np.concatenate((phase,(phase+1.0),(phase+2.0),(phase+3.0),(phase+4.0)))
	return(phase)

def find_smoothing_params(xphase):
	sorted_phases = np.sort(xphase)
	sorted_phases = np.concatenate((sorted_phases, sorted_phases+1.))
	sm = np.max(np.diff(sorted_phases))
	return(sm)
	

#def fit2(x1, y1, n, wt, mwt, c1, c2, c3, sigma_c1, sigma_c2, sigma_c3, chisum, q, jstep):
def fit_with_matrix(x1,y1,wt):
    sigma = 1.0 / wt
    sigma2 = sigma**2
    x2 = x1**2
    x3 = x1**3
    x4 = x1**4

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
    
    A = np.array([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])
    A_I = np.linalg.inv(A)
    M = np.array([C, E, G])
    R = np.dot(M, A_I)
    yest = R[0]
    return(yest)
	
