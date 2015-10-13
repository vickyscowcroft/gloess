#!/usr/bin/env/python

### September 18 2013
## Proposed change
## Rather than using the smoothing parameters as defined in the input file
## make a first guess in the program as 1./(N-1.) where N is the number of observations in that band
## This will catch the ones that need to be fit by a straight line
## Can add something to override this -- perhaps command line arguments to set a specific one?
## Need to make the database! This would be so much easier with everything together!!



import matplotlib
import numpy as np
import matplotlib.pyplot as mp
import sys
import gloess_fits as gf
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec

import os
os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin'
matplotlib.rc('text',usetex=True)
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Garamond']


def cm2inch(value):
    return value/2.54



du = []
db = []
dv = []
dr = []
di = []
dj = []
dh = []
dk = []
dir1 = []
dir2 = []
dir3 = []
dir4 = []
deu = []
deb = []
dev = []
der = []
dei = []
dej = []
deh = []
dek = []
deir1 = []
deir2 = []
deir3 = []
deir4 = []
dmjd = []


## Converting the gloess fortran/pgplot code to python/matplotlib
## June 15 2012

## Version 1.0
## last edit - June 19 2012

## Next thing to add:
##Print fits to an output text file

## Revision September 9 2013
## Making the phasing modular so it can have a different period for each wavelength
## This is to take into account when the data was taken


## Open the input data file and read the info

input = sys.argv[1]
counter = 0

## Want to know whether the IRAC data is phased or not. 
## If it is phased, must reduce the uncertainty by another factor of np.sqrt(N)
## if phased == 1 then true. if phased == 0, false

print input

for line in open(input):
	data = line.split()
	if counter == 0:	
		cepname = data[0]
	if counter == 1:
		pu = float(data[0])
		pb = float(data[1])
		pv = float(data[2])
		pr = float(data[3])
		pi = float(data[4])
		pj = float(data[5])
		ph = float(data[6])
		pk = float(data[7])
		pi1 = float(data[8])
		pi2 = float(data[9])
		pi3 = float(data[10])
		pi4 = float(data[10])
	if counter == 2:
		nlines = float(data[0])
	if counter == 3:
		xu = float(data[0])
		xb = float(data[1])
		xv = float(data[2])
		xr = float(data[3])
		xi = float(data[4])
		xj = float(data[5])
		xh = float(data[6])
		xk = float(data[7])
		xir1 = float(data[8])
		xir2 = float(data[9])
		xir3 = float(data[10])
		xir4 = float(data[11])
	if counter > 3:
		dmjd.append(float(data[0]))
		du.append(float(data[1]))
		deu.append(float(data[2]))
		db.append(float(data[3]))
		deb.append(float(data[4]))
		dv.append(float(data[5]))
		dev.append(float(data[6]))
		dr.append(float(data[7]))
		der.append(float(data[8]))
		di.append(float(data[9]))
		dei.append(float(data[10]))
		dj.append(float(data[11]))
		dej.append(float(data[12]))
		dh.append(float(data[13]))
		deh.append(float(data[14]))
		dk.append(float(data[15]))
		dek.append(float(data[16]))
		dir1.append(float(data[17]))
		deir1.append(float(data[18]))
		dir2.append(float(data[19]))
		deir2.append(float(data[20]))
		dir3.append(float(data[21]))
		deir3.append(float(data[22]))
		dir4.append(float(data[23]))
		deir4.append(float(data[24]))		
	counter  = counter + 1	
		
## Read in all the data from the file and filled the arrays. Need to convert these to numpy arrays.

number = counter - 4 # Number data lines in the file
#print number

u = np.array(du)
b = np.array(db)
v = np.array(dv)
r = np.array(dr)
i = np.array(di)
j = np.array(dj)
h = np.array(dh)
k = np.array(dk)
ir1 = np.array(dir1)
ir2 = np.array(dir2)
ir3 = np.array(dir3)
ir4 = np.array(dir4)
eu = np.array(deu)
eb = np.array(deb)
ev = np.array(dev)
er = np.array(der)
ei = np.array(dei)
ej = np.array(dej)
eh = np.array(deh)
ek = np.array(dek)
eir1 = np.array(deir1)
eir2 = np.array(deir2)
eir3 = np.array(deir3)
eir4 = np.array(deir4)
mjd = np.array(dmjd)

nu = sum(u<50)
nb = sum(b<50)
nv = sum(v<50)
nr = sum(r<50)
ni = sum(i<50)
nj = sum(j<50)
nh = sum(h<50)
nk = sum(k<50)
nir1 = sum(ir1<50)
nir2= sum(ir2<50)
nir3= sum(ir3<50)
nir4= sum(ir4<50)

# Usage:  fit_one_band(data,err,phases,n,smooth):
maxvals = []
minvals = []
if nu > 0:
	maxvals.append(np.amax(u[u<50])+3.0)
	minvals.append(np.amin(u[u<50])+3.0)
if nb > 0:
	maxvals.append(np.amax(b[b<50])+1.5)
	minvals.append(np.amin(b[b<50])+1.5)
if nv > 0:
	maxvals.append(np.amax(v[v<50])+1.2)
	minvals.append(np.amin(v[v<50])+1.2)
if nr > 0:
	maxvals.append(np.amax(r[r<50])+0.7)
	minvals.append(np.amin(r[r<50])+0.7)
if ni > 0:
	maxvals.append(np.amax(i[i<50])+0.2)
	minvals.append(np.amin(i[i<50])+0.2)
if nj > 0:
	maxvals.append(np.amax(j[j<50]))
	minvals.append(np.amin(j[j<50]))
if nh > 0:
	maxvals.append(np.amax(h[h<50])-0.4)
	minvals.append(np.amin(h[h<50])-0.4)
if nk > 0:
	maxvals.append(np.amax(k[k<50])-0.8)
	minvals.append(np.amin(k[k<50])-0.8)
if nir1 > 0:
	maxvals.append(np.amax(ir1[ir1<50])-1.4)
	minvals.append(np.amin(ir1[ir1<50])-1.4)
if nir2 > 0:
	maxvals.append(np.amax(ir2[ir2<50])-1.8)
	minvals.append(np.amin(ir2[ir2<50])-1.8)
if nir3 > 0:
	maxvals.append(np.amax(ir3[ir3<50])-2.2)
	minvals.append(np.amin(ir3[ir3<50])-2.2)
if nir4 > 0:
	maxvals.append(np.amax(ir4[ir4<50])-2.6)
	minvals.append(np.amin(ir4[ir4<50])-2.6)


maxvals = np.array(maxvals)
minvals = np.array(minvals)

max = np.max(maxvals)
min = np.min(minvals)
print cepname, ' ---- Period =', pi1, 'days'
print '------------------------------------------------------'

# Set up names for output files

#fitname = cepname + '.glo_fits'
avname = cepname + '.glo_avs'

avsout = open(avname,'w')
#fitout = open(fitname,'w')

maxlim = max + 0.5
minlim = min - 0.5



mp.clf()

#fig = mp.figure()
#ax1 = fig.add_subplot(111)
#mp.figure(figsize=(16.0,10.0))


gs = gridspec.GridSpec(3, 4)
ax1 = mp.subplot(gs[:, 0:2])
ax2 = mp.subplot(gs[0,2:4])
ax3 = mp.subplot(gs[1, 2:4])
ax4 = mp.subplot(gs[2, 2:4])
ax1.axis([1,3.5,(maxlim),(minlim)])
titlestring = cepname + ', P = ' + str(pi1) + ' days'
#print titlestring
mp.suptitle(titlestring, fontsize=20)

ax1.set_ylabel('Magnitude')
ax1.set_xlabel('Phase $\phi$')


## Fitting and plotting for each band
print nu, nb, nv, nr, ni, nj, nh, nk, nir1, nir2, nir3, nir4
if nu > 0:
	phaseu = gf.phase_mjds(pu,mjd)
	u1, ux, yu, yeu, xphaseu = gf.fit_one_band(u,eu,phaseu,nu,xu)
	ax1.plot(ux,u1+3.,'k-')
	ax1.plot(xphaseu,yu+3.,color='Violet',marker='o',ls='None', label='$U+3$')
	aveu, adevu, sdevu, varu, skewu, kurtosisu, ampu = gf.moment(u1[200:300],100)
	if nu > 1:
		print  'P = {3:.4f}   <U> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(aveu, sdevu/np.sqrt(nu), ampu, pu)
		print >> avsout, 'P = {3:.4f}   <U> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(aveu, sdevu/np.sqrt(nu), ampu, pu)
	if nu == 1:
		print  'P = {1:.4f}   U = {0:.3f} --- single point'.format(aveu, pu)
		print >> avsout, 'P = {1:.4f}   U = {0:.3f} --- single point'.format(aveu, pu)
		
if nb > 0:
	phaseb = gf.phase_mjds(pb,mjd)
	b1, bx, yb, yeb, xphaseb = gf.fit_one_band(b,eb,phaseb,nb,xb)
	ax1.plot(bx,b1+1.5,'k-')
	ax1.plot(xphaseb,yb+1.5,color='MediumSlateBlue',marker='o',ls='None', label='$B+1.5$')
	aveb, adevb, sdevb, varb, skewb, kurtosisb, ampb = gf.moment(b1[200:300],100)
	if nb > 1:
		print  'P = {3:.4f}   <B> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(aveb, sdevb/np.sqrt(nb), ampb, pb)
		print >> avsout, 'P = {3:.4f}   <B> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(aveb, sdevb/np.sqrt(nb), ampb, pb)
	if nb == 1:
		print  'P = {1:.4f}   B = {0:.3f} --- single point'.format(aveb, pb)
		print >> avsout, 'P = {1:.4f}   B = {0:.3f} --- single point'.format(aveb, pb)		

if nv > 0:
	phasev = gf.phase_mjds(pv,mjd)
	v1, vx, yv, yev, xphasev = gf.fit_one_band(v,ev,phasev,nv,xv)
	ax1.plot(vx,v1+1.2,'k-')
	ax1.plot(xphasev,yv+1.2,color='DodgerBlue',marker='o',ls='None', label='$V+1.2$')
	avev, adevv, sdevv, varv, skewv, kurtosisv, ampv = gf.moment(v1[200:300],100)
	if nv > 1:
		print  'P = {3:.4f}   <V> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(avev, sdevv/np.sqrt(nv), ampv, pv)
		print >> avsout, 'P = {3:.4f}   <V> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(avev, sdevv/np.sqrt(nv), ampv, pv)
	if nv == 1:
		print  'P = {1:.4f}   V = {0:.3f} --- single point'.format(avev, pv)
		print >> avsout, 'P = {1:.4f}   V = {0:.3f} --- single point'.format(avev, pv)		

if nr > 0:
	phaser = gf.phase_mjds(pr,mjd)
	r1, rx, yr, yer, xphaser = gf.fit_one_band(r,er,phaser,nr,xr)
	ax1.plot(rx,r1+0.7,'k-')
	ax1.plot(xphaser,yr+0.7,color='Turquoise',marker='o',ls='None', label='$R+0.7$')
	aver, adevr, sdevr, varr, skewr, kurtosisr, ampr = gf.moment(r1[200:300],100)
	if nr > 1:
		print  'P = {3:.4f}   <R> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(aver, sdevr/np.sqrt(nr), ampr, pr)
		print >> avsout, 'P = {3:.4f}   <R> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(aver, sdevr/np.sqrt(nr), ampr, pr)
	if nr == 1:
		print  'P = {1:.4f}   R = {0:.3f} --- single point'.format(aver, pr)
		print >> avsout, 'P = {1:.4f}   R = {0:.3f} --- single point'.format(aver, pr)		


if ni > 0:
	phasei = gf.phase_mjds(pi,mjd)
	i1, ix, yi, yei, xphasei = gf.fit_one_band(i,ei,phasei,ni,xi)
	ax1.plot(ix,i1+0.2,'k-')
	ax1.plot(xphasei,yi+0.2,color='LawnGreen',marker='o',ls='None', label='$I+0.2$')
	avei, adevi, sdevi, vari, skewi, kurtosisi, ampi = gf.moment(i1[200:300],100)
	if ni > 1:
		print  'P = {3:.4f}   <I> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(avei, sdevi/np.sqrt(ni), ampi, pi)
		print >> avsout, 'P = {3:.4f}   <I> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(avei, sdevi/np.sqrt(ni), ampi, pi)
	if ni == 1:
		print  'P = {1:.4f}   I = {0:.3f} --- single point'.format(avei, pi)
		print >> avsout, 'P = {1:.4f}   I = {0:.3f} --- single point'.format(avei, pi)		

if nj > 0:
	phasej = gf.phase_mjds(pj,mjd)
	j1, jx, yj, yej, xphasej = gf.fit_one_band(j,ej,phasej,nj,xj)
	ax1.plot(jx,j1,'k-')
	ax1.plot(xphasej,yj,color='Gold',marker='o',ls='None', label='$J$')
	avej, adevj, sdevj, varj, skewj, kurtosisj, ampj = gf.moment(j1[200:300],100)
	if nj > 1:
		print  'P = {3:.4f}   <J> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(avej, sdevj/np.sqrt(nj), ampj, pj)
		print >> avsout, 'P = {3:.4f}   <J> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(avej, sdevj/np.sqrt(nj), ampj, pj)
	if nj == 1:
		print  'P = {1:.4f}   J = {0:.3f} --- single point'.format(avej, pj)
		print >> avsout, 'P = {1:.4f}   J = {0:.3f} --- single point'.format(avej, pj)		

if nh > 0:
	phaseh = gf.phase_mjds(ph,mjd)
	h1, hx, yh, yeh, xphaseh = gf.fit_one_band(h,eh,phaseh,nh,xh)
	ax1.plot(hx,h1-0.4,'k-')
	ax1.plot(xphaseh,yh-0.4,color='DarkOrange',marker='o',ls='None', label='$H-0.4$')
	aveh, adevh, sdevh, varh, skewh, kurtosish, amph = gf.moment(h1[200:300],100)
	if nh > 1:
		print  'P = {3:.4f}   <H> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(aveh, sdevh/np.sqrt(nh), amph, ph)
		print >> avsout, 'P = {3:.4f}   <H> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(aveh, sdevh/np.sqrt(nh), amph, ph)
	if nh == 1:
		print  'P = {1:.4f}   H = {0:.3f} --- single point'.format(aveh, ph)
		print >> avsout, 'P = {1:.4f}   H = {0:.3f} --- single point'.format(aveh, ph)		

if nk > 0:
	phasek = gf.phase_mjds(pk,mjd)
	k1, kx, yk, yek, xphasek = gf.fit_one_band(k,ek,phasek,nk,xk)
	ax1.plot(kx,k1-0.8,'k-')
	ax1.plot(xphasek,yk-0.8,color='Red',marker='o',ls='None', label='$K-0.8$')
	avek, adevk, sdevk, vark, skewk, kurtosisk, ampk = gf.moment(k1[200:300],100)
	if nk > 1:
		print  'P = {3:.4f}   <K> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(avek, sdevk/np.sqrt(nk), ampk, pk)
		print >> avsout, 'P = {3:.4f}   <K> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(avek, sdevk/np.sqrt(nk), ampk, pk)
	if nk == 1:
		print  'P = {1:.4f}   K = {0:.3f} --- single point'.format(avek, pk)
		print >> avsout, 'P = {1:.4f}   K = {0:.3f} --- single point'.format(avek, pk)		

if nir1 > 0:
	phaseir1 = gf.phase_mjds(pi1,mjd)
	ir11, ir1x, yir1, yeir1, xphaseir1 = gf.fit_one_band(ir1,eir1,phaseir1,nir1,xir1)
	ax1.plot(ir1x,ir11-1.4,'k-')
	ax1.plot(xphaseir1,yir1-1.4,color='MediumVioletRed',marker='o',ls='None', label='$[3.6] - 1.4$')
	aveir1, adevir1, sdevir1, varir1, skewir1, kurtosisir1, ampi1 = gf.moment(ir11[200:300],100)
	if pi1 >= 12.:
		factor = np.sqrt(nir1)
	if pi1 < 12.:
		factor = 1 
	if nir1 > 1:
		print factor
		print >> avsout, 'P = {3:.4f}   <[3.6]> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}'.format(aveir1, sdevir1/(np.sqrt(nir1)*factor), ampi1,pi1)
		print  'P = {3:.4f}   <[3.6]> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(aveir1, sdevir1/(np.sqrt(nir1)*factor), ampi1,pi1)
	if nir1 == 1:
		print >> avsout, 'P = {1:.4f}   [3.6] = {0:.3f} --- single point'.format(aveir1,pi1)
		print  'P = {1:.4f}   [3.6] = {0:.3f} --- single point'.format(aveir1,pi1)

if nir2 > 0:
	phaseir2 = gf.phase_mjds(pi2,mjd)
	ir21, ir2x, yir2, yeir2, xphaseir2 = gf.fit_one_band(ir2,eir2,phaseir2,nir2,xir2)
	ax1.plot(ir2x,ir21-1.8,'k-')
	ax1.plot(xphaseir2,yir2-1.8,color='DeepPink',marker='o',ls='None', label='$[4.5] - 1.8$')
	aveir2, adevir2, sdevir2, varir2, skewir2, kurtosisir2, ampi2 = gf.moment(ir21[200:300],100)
	if pi2 >= 12.:
		factor = np.sqrt(nir2)
	if pi2 < 12.:
		factor = 1 
	if nir2 > 1:
		print >> avsout, 'P = {3:.4f}   <[4.5]> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}'.format(aveir2, sdevir2/(np.sqrt(nir2)*factor), ampi2,pi2)
		print  'P = {3:.4f}   <[4.5]> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(aveir2, sdevir2/(np.sqrt(nir2)*factor), ampi2,pi2)
	if nir2 == 1:
		print >> avsout, 'P = {1:.4f}   [4.5] = {0:.3f} --- single point'.format(aveir2,pi2)
		print  'P = {1:.4f}   [4.5] = {0:.3f} --- single point'.format(aveir2,pi2)

if nir3 > 0:
	phaseir3 = gf.phase_mjds(pi3,mjd)
	ir31, ir3x, yir3, yeir3, xphaseir3 = gf.fit_one_band(ir3,eir3,phaseir3,nir3,xir3)
	ax1.plot(ir3x,ir31-2.2,'k-')
	ax1.plot(xphaseir3,yir3-2.2,color='HotPink',marker='o',ls='None', label='$[5.8] - 2.2$')
	aveir3, adevir3, sdevir3, varir3, skewir3, kurtosisir3, ampi3 = gf.moment(ir31[200:300],100)
	if nir3 > 1:
		print  'P = {3:.4f}   <[5.8]> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(aveir3, sdevir3/np.sqrt(nir3), ampi3, pi3)
		print >> avsout, 'P = {3:.4f}   <[5.8]> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(aveir3, sdevir3/np.sqrt(nir3), ampi3, pi3)
	if nir3 == 1:
		print  'P = {1:.4f}   [5.8] = {0:.3f} --- single point'.format(aveir3, pi3)
		print >> avsout, 'P = {1:.4f}   [5.8] = {0:.3f} --- single point'.format(aveir3, pi3)		

if nir4 > 0:
	phaseir4 = gf.phase_mjds(pi4,mjd)
	ir41, ir4x, yir4, yeir4, xphaseir4 = gf.fit_one_band(ir4,eir4,phaseir4,nir4,xir4)
	ax1.plot(ir4x,ir41-2.6,'k-')
	ax1.plot(xphaseir4,yir4-2.6,color='PeachPuff',marker='o',ls='None', label='$[8.0] - 2.6$')
	aveir4, adevir4, sdevir4, varir4, skewir4, kurtosisir4, ampi4 = gf.moment(ir41[200:300],100)
	if nir4 > 1:
		print  'P = {3:.4f}   <[8.0]> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(aveir4, sdevir4/np.sqrt(nir4), ampi4, pi4)
		print >> avsout, 'P = {3:.4f}   <[8.0]> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(aveir4, sdevir4/np.sqrt(nir4), ampi4, pi4)
	if nir4 == 1:
		print  'P = {1:.4f}   [8.0] = {0:.3f} --- single point'.format(aveir4, pi4)
		print >> avsout, 'P = {1:.4f}   [8.0] = {0:.3f} --- single point'.format(aveir4, pi4)		



handles, labels = ax1.get_legend_handles_labels() 
#ax1.legend(handles[::-1],labels[::-1],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., numpoints=1)
ax1.legend(handles[::-1],labels[::-1],loc=4, numpoints=1,prop={'size':10})



#mp.setp(ax1.get_xticklabels(),visible=False)


### Define the colour curve
colour_curve = ir11 - ir21
## Define the colour points
ch1_points = yir1[yir1<99]
ch2_points = yir2[yir2<99]
colour_points = ch1_points - ch2_points
colour_phases = xphaseir1[yir1<99]

colour_points = np.concatenate((colour_points,colour_points,colour_points,colour_points,colour_points))
colour_phases = np.concatenate((colour_phases,(colour_phases+1.),(colour_phases+2.),(colour_phases+3.),(colour_phases+4.)))


avecol, adevcol, sdevcol, varcol, skewcol, kurtosiscol, ampcol = gf.moment(colour_curve[200:300],100)

print >> avsout, '<[3.6] - [4.5]> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(avecol, sdevcol/factor, ampcol)
print  '<[3.6] - [4.5]> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(avecol, sdevcol/factor, ampcol)

print np.average(ir11[200:300]) + 0.3
print np.average(ir11[200:300]) - 0.3

ax2.axis([1,3.5,(np.average(ir11[200:300]) + 0.4),(np.average(ir11[200:300]) - 0.4)])
ax2.yaxis.tick_right()
ax2.plot(ir1x,ir11,'k-')
ax2.plot(xphaseir1,yir1,color='MediumVioletRed',marker='o',ls='None', label='$[3.6]$')
ax2.annotate('$[3.6]$', xy=(0.04, 0.8375), xycoords='axes fraction', fontsize=16)

ax3.axis([1,3.5,(np.average(ir21[200:300]) + 0.4),(np.average(ir21[200:300]) - 0.4)])
ax3.yaxis.tick_right()
ax3.plot(ir2x,ir21,'k-')
ax3.plot(xphaseir2,yir2,color='DeepPink',marker='o',ls='None', label='$[3.6]$')
ax3.annotate('$[4.5]$', xy=(0.04, 0.8375), xycoords='axes fraction',fontsize=16)


#divider = make_axes_locatable(ax1)
#axcol = divider.append_axes("bottom",1.2,pad=0.1,sharex=ax1)
myaxis2 = [1,3.5,-0.2,0.2]
ax4.axis(myaxis2)
ax4.yaxis.tick_right()
ax4.yaxis.set_major_locator(mp.FixedLocator([-0.1,0,0.1]))
ax4.plot(ir1x,colour_curve,'k-')
ax4.plot(colour_phases,colour_points,color='Black',marker='o',ls='None', label='$[3.6]-[4.5]$')

ax4.set_xlabel('Phase $\phi$')
#ax4.annotate('$[3.6] - [4.5]$', xy=(1.1, 0.135), xycoords='data')
ax4.annotate('$[3.6] - [4.5]$', xy=(0.04, 0.8375), xycoords='axes fraction',fontsize=16)

ax4.hlines(0,1,3.5,'k','dashdot')

mp.setp(ax2.get_xticklabels(),visible=False)
mp.setp(ax3.get_xticklabels(),visible=False)

plotname = cepname+'.eps'
mp.savefig(plotname, transparent='True')

avsout.close()
#mp.show()
#fitout.close()
	
	

## Make publication plots

mp.clf()

#fig = mp.figure()
#ax1 = fig.add_subplot(111)
mp.figure(figsize=(5,5))

gs = gridspec.GridSpec(3, 1)

axp1 = mp.subplot(gs[0,0])
axp2 = mp.subplot(gs[1, 0])
axp3 = mp.subplot(gs[2, 0])



axp1.axis([1,3.5,(np.average(ir11[200:300]) + 0.4),(np.average(ir11[200:300]) - 0.4)])
#axp1.yaxis.tick_right()
axp1.plot(ir1x,ir11,'k-')
axp1.plot(xphaseir1,yir1,color='Black',marker='o',ls='None', label='$[3.6]$',ms=3)
axp1.yaxis.set_major_locator(mp.MultipleLocator(0.2))
axp1.annotate('$[3.6]$', xy=(0.04, 0.8375), xycoords='axes fraction', fontsize=12)


axp2.axis([1,3.5,(np.average(ir21[200:300]) + 0.4),(np.average(ir21[200:300]) - 0.4)])
#axp2.yaxis.tick_right()
axp2.plot(ir2x,ir21,'k-')
axp2.plot(xphaseir2,yir2,color='Black',marker='o',ls='None', label='$[4.5]$',ms=3)
axp2.yaxis.set_major_locator(mp.MultipleLocator(0.2))
axp2.annotate('$[4.5]$', xy=(0.04, 0.8375), xycoords='axes fraction', fontsize=12)

#divider = make_axes_locatable(ax1)
#axcol = divider.append_axes("bottom",1.2,pad=0.1,sharex=ax1)
myaxis2 = [1,3.5,-0.2,0.2]
axp3.axis(myaxis2)
#axp3.yaxis.tick_right()
axp3.yaxis.set_major_locator(mp.FixedLocator([-0.1,0,0.1]))
axp3.plot(ir1x,colour_curve,'k-')
axp3.plot(colour_phases,colour_points,color='Black',marker='o',ls='None', label='$[3.6]-[4.5]$',ms=3)

axp3.set_xlabel('Phase $\phi$')
#axp3.annotate('$[3.6] - [4.5]$', xy=(1.1, 0.135), xycoords='data')
axp3.annotate('$[3.6] - [4.5]$', xy=(0.04, 0.8375), xycoords='axes fraction',fontsize=12)
axp3.hlines(0,1,3.5,'k','dashdot')

mp.setp(axp1.get_xticklabels(),visible=False)
mp.setp(axp2.get_xticklabels(),visible=False)

#titlestring = cepname + ', P = ' + str(pi1) + ' days'
titlestring = '{0}, P = {1:.3f} days'.format(cepname, pi1)
#print titlestring
mp.suptitle(titlestring, fontsize=14)


plotname_pub = cepname+'_pub.eps'
mp.savefig(plotname_pub, transparent='True')

															
		

		
	


