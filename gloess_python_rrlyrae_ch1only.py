#!/usr/bin/env/python

import numpy as np
import matplotlib.pyplot as mp
import sys
import gloess_fits as gf
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin'
from matplotlib import rcParams
matplotlib.rc('text',usetex=True)

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
	min = np.amin(ir1[ir1<50])



print cepname, ' ---- Period =', period, 'days'
print '------------------------------------------------------'

# Set up names for output files

fitname = cepname + '.glo_fits'
avname = cepname + '.glo_avs'

avsout = open(avname,'w')
fitout = open(fitname,'w')

maxlim = max + 0.2
minlim = min - 0.2
mp.close()
mp.clf()
mp.axis([1,3.5,(maxlim),(minlim)])
ax1 = subplot(111)
#mp.xlabel('Phase $\phi$')
mp.ylabel('[3.6]')
titlestring = cepname + ', P = ' + str(period) + ' days'
#print titlestring
mp.title(titlestring)


## Fitting and plotting for each band


if nir1 > 0:
	ir11, ir1x, yir1, yeir1, xphaseir1 = gf.fit_one_band(ir1,eir1,phase,nir1,xir1)
#	ax1.plot(ir1x,ir11-0.9,'k-')
# 	ax1.plot(xphaseir1,yir1-0.9,color='MediumVioletRed',marker='o',ls='None', label='[3.6]-0.9')
## for RRLyrae WISE plots:
	#mag1string = '<[3.6]> = ' + str(aveir1) + ' $\pm$ ' + str(sdevir1)
	ax1.plot(ir1x,ir11,'k-')
	ax1.errorbar(xphaseir1, yir1, yeir1, color='k', ls='None') 
 	ax1.plot(xphaseir1,yir1,color='Turquoise',marker='o',ls='None', label='[3.6]')
	aveir1, adevir1, sdevir1, varir1, skewir1, kurtosisir1, ampir1 = gf.moment(ir11[200:300],100)
	if phased == 1:
		factor = sqrt(nir1)
	if phased == 0:
		factor = 1
	if nir1 > 1:
		print >> avsout, '<[3.6]> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f} N ir1 = {3}'.format(aveir1, sdevir1/factor, ampir1,nir1)
		print  '<[3.6]> = {0:.3f}    std dev = {1:.3f}     amplitude = {2:.3f}' .format(aveir1, sdevir1/factor, ampir1)
	if nir1 == 1:
		print >> avsout, '[3.6] = {0:.3f} --- single point'.format(aveir1)
		print  '[3.6] = {0:.3f} --- single point'.format(aveir1)


handles, labels = ax1.get_legend_handles_labels() 
#ax1.legend(handles[::-1],labels[::-1],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., numpoints=1)
#ax1.legend(handles[::-1],labels[::-1],loc=1, numpoints=1, fancybox=True)


annotation = '$[3.6] = {0:.3f} \pm {1:.3f}$ '.format(aveir1,sdevir1/factor)
ax1.annotate(annotation,xy=(1.55,max + 0.12), xycoords='data', ha='center', size=16, bbox=dict(boxstyle="round",fc='w'))
#ax1.annotate(annotation, xy=(0.5, 0.5), xycoords='axes fraction', ha='center', bbox=dict(boxstyle="round",fc='w'))

mp.xlabel('Phase $\phi$')

mp.show()


plotname = cepname+'.pdf'
mp.savefig(plotname, transparent='True')

avsout.close()

#fitout.close()
	
	







															
		

		
	


