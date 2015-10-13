#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as mp
import sys
import gloess_fits as gf
import os
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin'
matplotlib.rc('text',usetex=True)
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Garamond']



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
di3 = []
di4 = []
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
dei3 = []
dei4 = []
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
phased = int(sys.argv[2])
wantColor = int(sys.argv[3])

for line in open(input):
	data = line.split()
	if counter == 0:	
		cepname = data[0]
	if counter == 1:
		period = float(data[0])
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
		xi3 = float(data[10])
		xi4 = float(data[11])
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
		di3.append(float(data[21]))
		dei3.append(float(data[22]))
		di4.append(float(data[23]))
		dei4.append(float(data[24]))		
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
i3 = np.array(di3)
i4 = np.array(di4)
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
ei3 = np.array(dei3)
ei4 = np.array(dei4)
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
ni3= sum(i3<50)
ni4= sum(i4<50)

# Phases don't need to be done individually by band - only depends on P
phase = (mjd / period) - np.floor(mjd / period)
phase = np.concatenate((phase,(phase+1.0),(phase+2.0),(phase+3.0),(phase+4.0)))

# Usage:  fit_one_band(data,err,phases,n,smooth):
if nv > 0:
	max = np.amax(v[v<50])
	min = np.amax(v[v<50])
elif ni > 0:
	max = np.amax(i[i<50])
	min = np.amax(i[i<50])
elif nj > 0:
	max = np.amax(j[j<50])
	min = np.amax(j[j<50])
elif nir1 > 0:
	max = np.amax(ir1[ir1<50])
	min = np.amax(ir1[ir1<50])



print cepname, ' ---- Period =', period, 'days'
print '------------------------------------------------------'

# Set up names for output files

fitname = cepname + '.glo_fits'
avname = cepname + '.glo_avs'

avsout = open(avname,'w')
fitout = open(fitname,'w')

maxlim = max + 0.3
minlim = min - 1.2

mp.clf()
mp.axis([1,3.5,(maxlim),(minlim)])
ax1 = mp.subplot(111)
#mp.xlabel('Phase $\phi$')
mp.ylabel('Magnitude')
titlestring = cepname + ', P = ' + str(period) + ' days'
#print titlestring
mp.title(titlestring)


## Fitting and plotting for each band

if nu > 0:
	u1, ux, yu, yeu, xphaseu = gf.fit_one_band(u,eu,phase,nu,xu)
	ax1.plot(ux,u1+3.,'k-')
	ax1.plot(xphaseu,yu+3.,color='Violet',marker='o',ls='None', label='U+3')
	aveu, adevu, sdevu, varu, skewu, kurtosisu, ampu = gf.moment(u1[200:300],100)
	if nu > 1:
		print  '<U> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(aveu, sdevu, ampu)
		print >> avsout, '<U> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(aveu, sdevu, ampu)
	if nu == 1:
		print  'U = {0:.3f} --- single point'.format(aveu)
		print >> avsout, 'U = {0:.3f} --- single point'.format(aveu)
		
if nb > 0:
	b1, bx, yb, yeb, xphaseb = gf.fit_one_band(b,eb,phase,nb,xb)
	ax1.plot(bx,b1+1.5,'k-')
	ax1.plot(xphaseb,yb+1.5,color='MediumSlateBlue',marker='o',ls='None', label='B+1.5')
	aveb, adevb, sdevb, varb, skewb, kurtosisb, ampb = gf.moment(b1[200:300],100)
	if nb > 1:
		print  '<B> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(aveb, sdevb, ampb)
		print >> avsout, '<B> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(aveb, sdevb, ampb)
	if nb == 1:
		print  'B = {0:.3f} --- single point'.format(aveb)
		print >> avsout,  'B = {0:.3f} --- single point'.format(aveb)
		
if nv > 0:
	v1, vx, yv, yev, xphasev = gf.fit_one_band(v,ev,phase,nv,xv)
	ax1.plot(vx,v1+1.2,'k-')
	ax1.plot(xphasev,yv+1.2,color='DodgerBlue',marker='o',ls='None', label='V+1.2')
	avev, adevv, sdevv, varv, skewv, kurtosisv, ampv = gf.moment(v1[200:300],100)
	if nv > 1:
		print >> avsout, '<V> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f} ' .format(avev, sdevv, ampv)
		print  '<V> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f} ' .format(avev, sdevv, ampv)
	if nv == 1:
		print  'V = {0:.3f} --- single point'.format(avev)
		print   >> avsout, 'V = {0:.3f} --- single point'.format(avev)

if nr > 0:
	r1, rx, yr, yer, xphaser = gf.fit_one_band(r,er,phase,nr,xr)
	ax1.plot(rx,r1+0.7,'k-')
	ax1.plot(xphaser,yr+0.7,color='Turquoise',marker='o',ls='None', label='R+0.7')
	aver, adevr, sdevr, varr, skewr, kurtosisr, ampr = gf.moment(r1[200:300],100)
	if nr > 1:
		print '<R> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(aver, sdevr, ampr)
		print >> avsout, '<R> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(aver, sdevr, ampr)
	if nr == 1:
		print   >> avsout, 'R = {0:.3f} --- single point'.format(aver)
		print    'R = {0:.3f} --- single point'.format(aver)
	
if ni > 0:
	i1, ix, yi, yei, xphasei = gf.fit_one_band(i,ei,phase,ni,xi)
	ax1.plot(ix,i1+0.2,'k-')
	ax1.plot(xphasei,yi+0.2,color='LawnGreen',marker='o',ls='None', label='I+0.2')
	avei, adevi, sdevi, vari, skewi, kurtosisi, ampi = gf.moment(i1[200:300],100)
	if ni > 1:
		print  '<I> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(avei, sdevi, ampi)
		print >> avsout, '<I> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(avei, sdevi, ampi)
	if ni == 1:
		print   >> avsout, 'I = {0:.3f} --- single point'.format(avei)
		print   'I = {0:.3f} --- single point'.format(avei)

	
if nj > 0:
	j1, jx, yj, yej, xphasej = gf.fit_one_band(j,ej,phase,nj,xj)
	ax1.plot(jx,j1,'k-')
	ax1.plot(xphasej,yj,color='Gold',marker='o',ls='None', label='J')
	avej, adevj, sdevj, varj, skewj, kurtosisj, ampj = gf.moment(j1[200:300],100)
	if nj > 1:
		print >> avsout, '<J> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(avej, sdevj, ampj)
		print '<J> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(avej, sdevj, ampj)
	if nj == 1:
		print   >> avsout, 'J = {0:.3f} --- single point'.format(avej)
		print  'J = {0:.3f} --- single point'.format(avej)
	
if nh > 0:
	h1, hx, yh, yeh, xphaseh = gf.fit_one_band(h,eh,phase,nh,xh)
	ax1.plot(hx,h1-0.3,'k-')
	ax1.plot(xphaseh,yh-0.3,color='DarkOrange',marker='o',ls='None', label='H-0.3')
	aveh, adevh, sdevh, varh, skewh, kurtosish, amph = gf.moment(h1[200:300],100)
	if nh > 1:
		print >> avsout, '<H> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(aveh, sdevh, amph)
		print  '<H> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(aveh, sdevh, amph)
	if nh == 1:
		print >> avsout, 'H = {0:.3f} --- single point'.format(aveh)
		print 'H = {0:.3f} --- single point'.format(aveh)

if nk > 0:
	k1, kx, yk, yek, xphasek = gf.fit_one_band(k,ek,phase,nk,xk)
	ax1.plot(kx,k1-0.6,'k-')
	ax1.plot(xphasek,yk-0.6,color='Red',marker='o',ls='None', label='K-0.6')
	avek, adevk, sdevk, vark, skewk, kurtosisk, ampk = gf.moment(k1[200:300],100)
	if nk > 1:
		print >> avsout, '<K> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(avek, sdevk, ampk)
		print  '<K> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(avek, sdevk, ampk)
	if nk == 1:
		print >> avsout, 'K = {0:.3f} --- single point'.format(avek)
		print  'K = {0:.3f} --- single point'.format(avek)

if nir1 > 0:
	ir11, ir1x, yir1, yeir1, xphaseir1 = gf.fit_one_band(ir1,eir1,phase,nir1,xir1)
#	ax1.plot(ir1x,ir11-0.9,'k-')
# 	ax1.plot(xphaseir1,yir1-0.9,color='MediumVioletRed',marker='o',ls='None', label='[3.6]-0.9')
## for RRLyrae WISE plots:
	#mag1string = '<[3.6]> = ' + str(aveir1) + ' $\pm$ ' + str(sdevir1)
	ax1.plot(ir1x,ir11,'k-')
 	ax1.errorbar(xphaseir1, yir1, yerr=yeir1, color='k', ls='None')
 	ax1.plot(xphaseir1,yir1,color='Turquoise',marker='o',ls='None', label='[3.6]')
	aveir1, adevir1, sdevir1, varir1, skewir1, kurtosisir1, ampir1 = gf.moment(ir11[200:300],100)
	if phased == 1:
		factor = np.sqrt(nir1)
	if phased == 0:
		factor = 1
	if nir1 > 1:
		print >> avsout, '<[3.6]> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f} N ir1 = {3}'.format(aveir1, sdevir1/factor, ampir1,nir1)
		print  '<[3.6]> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(aveir1, sdevir1/factor, ampir1)
	if nir1 == 1:
		print >> avsout, '[3.6] = {0:.3f} --- single point'.format(aveir1)
		print  '[3.6] = {0:.3f} --- single point'.format(aveir1)

if nir2 > 0:
	ir21, ir2x, yir2, yeir2, xphaseir2 = gf.fit_one_band(ir2,eir2,phase,nir2,xir2)
#	ax1.plot(ir2x,ir21-1.2,'k-')
# 	ax1.plot(xphaseir2,yir2-1.2,color='DeepPink',marker='o',ls='None', label='[4.5]-1.2')
## For RRLyrae WISE plots:
	#mag2string = '<[4.5]> = ' + str(aveir2) + ' $\pm$ ' + str(sdevir2)
	ax1.plot(ir2x,ir21-0.4,'k-')
	ax1.errorbar(xphaseir2, yir2-0.4, yerr=yeir2, color='k', ls='None')
 	ax1.plot(xphaseir2,yir2-0.4,color='Gold',marker='o',ls='None', label='[4.5] - 0.4')
	aveir2, adevir2, sdevir2, varir2, skewir2, kurtosisir2, ampir2= gf.moment(ir21[200:300],100)
	if phased == 1:
		factor = np.sqrt(nir2)
	if phased == 0:
		factor = 1
	if nir2 > 1:
		print >> avsout, '<[4.5]> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f} N IR2 = {3}'.format(aveir2, sdevir2/factor, ampir2,nir2)
		print  '<[4.5]> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(aveir2, sdevir2/factor, ampir2)
	if nir2 == 1:
		print >> avsout, '[4.5] = {0:.3f} --- single point'.format(aveir2)
		print  '[4.5] = {0:.3f} --- single point'.format(aveir2)

if ni3 > 0:
	i31, i3x, yi3, yei3, xphasei3 = gf.fit_one_band(i3,ei3,phase,ni3,xi3)
#	ax1.plot(i3x,i31-1.5,'k-')
# 	ax1.plot(xphasei3,yi3-1.5,color='HotPink',marker='o',ls='None', label='[5.8]-1.5')
## For RRRLyrae WISE plots:
	ax1.plot(i3x,i31-1.,'k-')
 	ax1.plot(xphasei3,yi3-1.,color='DeepPink',marker='o',ls='None', label='[5.8]-1.0')
	avei3, adevi3, sdevi3, vari3, skewi3, kurtosisi3, ampi3 = gf.moment(i31[200:300],100)
	if ni3 > 1:
		print >> avsout, '<[5.8]> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(avei3, sdevi3, ampi3)
		print  '<[5.8]> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(avei3, sdevi3, ampi3)
	if ni3 == 1:
		print >> avsout, '[5.8] = {0:.3f} --- single point'.format(avei3)
		print  '[5.8] = {0:.3f} --- single point'.format(avei3)

if ni4 > 0:
	i41, i4x, yi4, yei4, xphasei4 = gf.fit_one_band(i4,ei4,phase,ni4,xi4)
	ax1.plot(i4x,i41-1.8,'k-')
 	ax1.plot(xphasei4,yi4-1.8,color='PeachPuff',marker='o',ls='None', label='[8.0]-1.8')
	avei4, adevi4, sdevi4, vari4, skewi4, kurtosisi4, ampi4 = gf.moment(i41[200:300],100)
	if ni4 > 1:
		print >> avsout, '<[8.0]> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(avei4, sdevi4, ampi4)
		print  '<[8.0]> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(avei4, sdevi4, ampi4)
	if ni4 == 1:
		print >> avsout, '[8.0] = {0:.3f} --- single point'.format(avei4)
		print  '[8.0] = {0:.3f} --- single point'.format(avei4)

handles, labels = ax1.get_legend_handles_labels() 
#ax1.legend(handles[::-1],labels[::-1],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., numpoints=1)
ax1.legend(handles[::-1],labels[::-1],loc=1, numpoints=1, fancybox=True)


annotation = '$[3.6] = {0:.3f} \pm {1:.3f}$ \n \n $[4.5] = {2:.3f} \pm {3:.3f}$'.format(aveir1,sdevir1/factor,aveir2,sdevir2/factor)
ax1.annotate(annotation,xy=(1.37,min-0.92), xycoords='data', ha='center', bbox=dict(boxstyle="round",fc='w'))

if wantColor == 1:

	ax1.legend(handles[::-1],labels[::-1],loc=1, numpoints=1)
	mp.setp(ax1.get_xticklabels(),visible=False)

	colour = ir11 - ir21

	divider = make_axes_locatable(ax1)
	axcorr = divider.append_axes("bottom",1.2,pad=0.1,sharex=ax1)
	myaxis2 = [1.0,3.5,-0.2,0.2]
	axcorr.axis(myaxis2)
	axcorr.yaxis.set_major_locator(plt.FixedLocator([-0.1,0.0,0.1]))
	axcorr.plot(ir1x,colour,'k-')
	mp.ylabel('[3.6] - [4.5]')

mp.xlabel('Phase $\phi$')

mp.show()

for count in range(200,300):
	print >> fitout, '{0:.2f} {1:.3f} {2:.3f}'.format(ir1x[count]-1.0, ir11[count], ir21[count])


plotname = cepname+'.eps'
mp.savefig(plotname, transparent='True')

avsout.close()



fitout.close()
	
	







															
		

		
	


