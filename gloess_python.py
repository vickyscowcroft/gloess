#!/usr/bin/env/python

import numpy as np
import matplotlib.pyplot as mp
import sys
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import os
os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin'
matplotlib.rc('text',usetex=True)
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Garamond']

sys.path.append('/Users/vs522/Dropbox/Python')

import gloess_fits as gf


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


## Converting the gloess fourtran/pgplot code to python/matplotlib
## June 15 2012

## Version 1.0
## last edit - June 19 2012

## Next thing to add:
##Print fits to an output text file


## Open the input data file and read the info

input = sys.argv[1]
counter = 0

## Want to know whether the IRAC data is phased or not. 
## If it is phased, must reduce the uncertainty by another factor of sqrt(N)
## if phased == 1 then true. if phased == 0, false

print input

for line in open(input):
	data = line.split()
	if counter == 0:	
		cepname = data[0]
	if counter == 1:
		period = float(data[0])
		if period > 0:
			phased = 1
		else:
			phased = 0
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

# Phases don't need to be done individually by band - only depends on P
phase = (mjd / period) - np.floor(mjd / period)
phase = np.concatenate((phase,(phase+1.0),(phase+2.0),(phase+3.0),(phase+4.0)))

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
print cepname, ' ---- Period =', period, 'days'
print '------------------------------------------------------'

# Set up names for output files

#fitname = cepname + '.glo_fits'
avname = cepname + '.glo_avs'

avsout = open(avname,'w')
#fitout = open(fitname,'w')

maxlim = max + 0.5
minlim = min - 0.5



mp.clf()

#fig = plt.figure()
#ax1 = fig.add_subplot(111)
#mp.figure(figsize=(16.0,10.0))


gs = gridspec.GridSpec(3, 4)
ax1 = plt.subplot(gs[:, 0:2])
ax2 = plt.subplot(gs[0,2:4])
ax3 = plt.subplot(gs[1, 2:4])
ax4 = plt.subplot(gs[2, 2:4])
ax1.axis([1,3.5,(maxlim),(minlim)])
titlestring = cepname + ', P = ' + str(period) + ' days'
#print titlestring
mp.suptitle(titlestring, fontsize=20)

ax1.set_ylabel('Magnitude')
ax1.set_xlabel('Phase $\phi$')


## Fitting and plotting for each band
print nu, nb, nv, nr, ni, nj, nh, nk, nir1, nir2, nir3, nir4
if nu > 0:
	u1, ux, yu, yeu, xphaseu = gf.fit_one_band(u,eu,phase,nu,xu)
	ax1.plot(ux,u1+3.,'k-')
	ax1.plot(xphaseu,yu+3.,color='Violet',marker='o',ls='None', label='$U+3$')
	aveu, adevu, sdevu, varu, skewu, kurtosisu, ampu = gf.moment(u1[200:300],100)
	if nu > 1:
		print  '<U> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(aveu, sdevu/sqrt(nu), ampu)
		print >> avsout, '<U> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(aveu, sdevu/sqrt(nu), ampu)
	if nu == 1:
		print  'U = {0:.3f} --- single point'.format(aveu)
		print >> avsout, 'U = {0:.3f} --- single point'.format(aveu)
		
if nb > 0:
	b1, bx, yb, yeb, xphaseb = gf.fit_one_band(b,eb,phase,nb,xb)
	ax1.plot(bx,b1+1.5,'k-')
	ax1.plot(xphaseb,yb+1.5,color='MediumSlateBlue',marker='o',ls='None', label='$B+1.5$')
	aveb, adevb, sdevb, varb, skewb, kurtosisb, ampb = gf.moment(b1[200:300],100)
	if nb > 1:
		print  '<B> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(aveb, sdevb/sqrt(nb), ampb)
		print >> avsout, '<B> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(aveb, sdevb/sqrt(nb), ampb)
	if nb == 1:
		print  'B = {0:.3f} --- single point'.format(aveb)
		print >> avsout,  'B = {0:.3f} --- single point'.format(aveb)
		
if nv > 0:
	v1, vx, yv, yev, xphasev = gf.fit_one_band(v,ev,phase,nv,xv)
	ax1.plot(vx,v1+1.2,'k-')
	ax1.plot(xphasev,yv+1.2,color='DodgerBlue',marker='o',ls='None', label='$V+1.2$')
	avev, adevv, sdevv, varv, skewv, kurtosisv, ampv = gf.moment(v1[200:300],100)
	if nv > 1:
		print >> avsout, '<V> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f} ' .format(avev, sdevv/sqrt(nv), ampv)
		print  '<V> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f} ' .format(avev, sdevv/sqrt(nv), ampv)
	if nv == 1:
		print  'V = {0:.3f} --- single point'.format(avev)
		print   >> avsout, 'V = {0:.3f} --- single point'.format(avev)

if nr > 0:
	r1, rx, yr, yer, xphaser = gf.fit_one_band(r,er,phase,nr,xr)
	ax1.plot(rx,r1+0.7,'k-')
	ax1.plot(xphaser,yr+0.7,color='Turquoise',marker='o',ls='None', label='$R+0.7$')
	aver, adevr, sdevr, varr, skewr, kurtosisr, ampr = gf.moment(r1[200:300],100)
	if nr > 1:
		print '<R> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(aver, sdevr/sqrt(nr), ampr)
		print >> avsout, '<R> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(aver, sdevr/sqrt(nr), ampr)
	if nr == 1:
		print   >> avsout, 'R = {0:.3f} --- single point'.format(aver)
		print    'R = {0:.3f} --- single point'.format(aver)
	
if ni > 0:
	i1, ix, yi, yei, xphasei = gf.fit_one_band(i,ei,phase,ni,xi)
	ax1.plot(ix,i1+0.2,'k-')
	ax1.plot(xphasei,yi+0.2,color='LawnGreen',marker='o',ls='None', label='$I+0.2$')
	avei, adevi, sdevi, vari, skewi, kurtosisi, ampi = gf.moment(i1[200:300],100)
	if ni > 1:
		print  '<I> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(avei, sdevi/sqrt(ni), ampi)
		print >> avsout, '<I> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(avei, sdevi/sqrt(ni), ampi)
	if ni == 1:
		print   >> avsout, 'I = {0:.3f} --- single point'.format(avei)
		print   'I = {0:.3f} --- single point'.format(avei)

	
if nj > 0:
	j1, jx, yj, yej, xphasej = gf.fit_one_band(j,ej,phase,nj,xj)
	ax1.plot(jx,j1,'k-')
	ax1.plot(xphasej,yj,color='Gold',marker='o',ls='None', label='$J$')
	avej, adevj, sdevj, varj, skewj, kurtosisj, ampj = gf.moment(j1[200:300],100)
	if nj > 1:
		print >> avsout, '<J> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(avej, sdevj/sqrt(nj), ampj)
		print '<J> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(avej, sdevj/sqrt(nj), ampj)
	if nj == 1:
		print   >> avsout, 'J = {0:.3f} --- single point'.format(avej)
		print  'J = {0:.3f} --- single point'.format(avej)
	
if nh > 0:
	h1, hx, yh, yeh, xphaseh = gf.fit_one_band(h,eh,phase,nh,xh)
	ax1.plot(hx,h1-0.4,'k-')
	ax1.plot(xphaseh,yh-0.4,color='DarkOrange',marker='o',ls='None', label='$H-0.4$')
	aveh, adevh, sdevh, varh, skewh, kurtosish, amph = gf.moment(h1[200:300],100)
	if nh > 1:
		print >> avsout, '<H> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(aveh, sdevh/sqrt(nh), amph)
		print  '<H> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(aveh, sdevh/sqrt(nh), amph)
	if nh == 1:
		print >> avsout, 'H = {0:.3f} --- single point'.format(aveh)
		print 'H = {0:.3f} --- single point'.format(aveh)

if nk > 0:
	k1, kx, yk, yek, xphasek = gf.fit_one_band(k,ek,phase,nk,xk)
	ax1.plot(kx,k1-0.8,'k-')
	ax1.plot(xphasek,yk-0.8,color='Red',marker='o',ls='None', label='$K-0.8$')
	avek, adevk, sdevk, vark, skewk, kurtosisk, ampk = gf.moment(k1[200:300],100)
	if nk > 1:
		print >> avsout, '<K> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(avek, sdevk/sqrt(nk), ampk)
		print  '<K> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(avek, sdevk/sqrt(nk), ampk)
	if nk == 1:
		print >> avsout, 'K = {0:.3f} --- single point'.format(avek)
		print  'K = {0:.3f} --- single point'.format(avek)

if nir1 > 0:
	ir11, ir1x, yir1, yeir1, xphaseir1 = gf.fit_one_band(ir1,eir1,phase,nir1,xir1)
	ax1.plot(ir1x,ir11-1.4,'k-')
 	ax1.plot(xphaseir1,yir1-1.4,color='MediumVioletRed',marker='o',ls='None', label='$[3.6]-1.4$')
## for RRLyrae WISE plots:
#	ax1.plot(ir1x,ir11+1.,'k-')
# 	ax1.plot(xphaseir1,yir1+1.,color='Turquoise',marker='o',ls='None', label='W1+1.0')
	aveir1, adevir1, sdevir1, varir1, skewir1, kurtosisir1, ampir1 = gf.moment(ir11[200:300],100)
	if phased == 1:
		factor = sqrt(nir1)
	if phased == 0:
		factor = 1 
	if nir1 > 1:
		print >> avsout, '<[3.6]> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f} N I1 = {3}'.format(aveir1, sdevir1/factor, ampir1,nir1)
		print  '<[3.6]> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(aveir1, sdevir1/factor, ampir1)
	if nir1 == 1:
		print >> avsout, '[3.6] = {0:.3f} --- single point'.format(aveir1)
		print  '[3.6] = {0:.3f} --- single point'.format(aveir1)

if nir2 > 0:
	ir21, ir2x, yir2, yeir2, xphaseir2 = gf.fit_one_band(ir2,eir2,phase,nir2,xir2)
	ax1.plot(ir2x,ir21-1.8,'k-')
 	ax1.plot(xphaseir2,yir2-1.8,color='DeepPink',marker='o',ls='None', label='$[4.5]-1.8$')
## For RRLyrae WISE plots:
#	ax1.plot(ir2x,ir21,'k-')
# 	ax1.plot(xphaseir2,yir2,color='Gold',marker='o',ls='None', label='W2')
	aveir2, adevir2, sdevir2, varir2, skewir2, kurtosisir2, ampir2= gf.moment(ir21[200:300],100)
	if phased == 1:
		factor = sqrt(nir2)
	if phased == 0:
		factor = 1

	if nir2 > 1:
		print >> avsout, '<[4.5]> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f} N I2 = {3}' .format(aveir2, sdevir2/factor, ampir2,nir2)
		print '<[4.5]> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(aveir2, sdevir2/factor, ampir2)
	if nir2 == 1:
		print >> avsout, '[4.5] = {0:.3f} --- single point'.format(aveir2)
		print '[4.5] = {0:.3f} --- single point'.format(aveir2)

if nir3 > 0:
	ir31, ir3x, yir3, yeir3, xphaseir3 = gf.fit_one_band(ir3,eir3,phase,nir3,xir3)
	ax1.plot(ir3x,ir31-2.2,'k-')
 	ax1.plot(xphaseir3,yir3-2.2,color='HotPink',marker='o',ls='None', label='$[5.8]-2.2$')
## For RRRLyrae WISE plots:
#	ax1.plot(ir3x,ir31-1.,'k-')
# 	ax1.plot(xphaseir3,yir3-1.,color='DeepPink',marker='o',ls='None', label='W3-1.0')
	aveir3, adevir3, sdevir3, varir3, skewir3, kurtosisir3, ampir3 = gf.moment(ir31[200:300],100)
	if nir3 > 1:
		print >> avsout, '<[5.8]> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(aveir3, sdevir3, ampir3)
		print  '<[5.8]> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(aveir3, sdevir3, ampir3)
	if nir3 == 1:
		print >> avsout, '[5.8] = {0:.3f} --- single point'.format(aveir3)
		print  '[5.8] = {0:.3f} --- single point'.format(aveir3)

if nir4 > 0:
	ir41, ir4x, yir4, yeir4, xphaseir4 = gf.fit_one_band(ir4,eir4,phase,nir4,xir4)
	ax1.plot(ir4x,ir41-2.6,'k-')
 	ax1.plot(xphaseir4,yir4-2.6,color='PeachPuff',marker='o',ls='None', label='$[8.0]-2.6$')
	aveir4, adevir4, sdevir4, varir4, skewir4, kurtosisir4, ampir4 = gf.moment(ir41[200:300],100)
	if nir4 > 1:
		print >> avsout, '<[8.0]> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(aveir4, sdevir4, ampir4)
		print  '<[8.0]> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(aveir4, sdevir4, ampir4)
	if nir4 == 1:
		print >> avsout, '[8.0] = {0:.3f} --- single point'.format(aveir4)
		print  '[8.0] = {0:.3f} --- single point'.format(aveir4)

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
ax4.yaxis.set_major_locator(plt.FixedLocator([-0.1,0,0.1]))
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
mp.show()
#fitout.close()
	
	







															
		

		
	


