#!/usr/bin/env/python

import numpy as np
import matplotlib.pyplot as mp
import sys
import gloess_fits as gf
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import os
import re

from matplotlib import rcParams
rcParams['font.size']=16

shift = float(sys.argv[2])

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
## If it is phased, must reduce the uncertainty by another factor of np.sqrt(N)
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


## gloess differential

v1, vx, yv, yev, xphasev = gf.fit_one_band(v,ev,phase,nv,xv)
ir11, ir1x, yir1, yeir1, xphaseir1 = gf.fit_one_band(ir1,eir1,phase,nir1,xir1)

avev, adevv, sdevv, varv, skewv, kurtosisv, ampv = gf.moment(v1[200:300],100)
aveir1, adevir1, sdevir1, varir1, skewir1, kurtosisir1, ampir1 = gf.moment(ir11[200:300],100)

offset = avev - aveir1

off_ir11 = ir11 + offset
off_yir1 = yir1 + offset

dy1dp_max_light = 1000
dy1dp_min_light = 1000

dyvdp_max_light = 1000
dyvdp_min_light = 1000

## Find min phase for ir and v

for ptop in range(301,350):
	dp = 0.01
	dy1 = ir11[ptop] - ir11[ptop - 1]
	dyv = v1[ptop] - v1[ptop - 2]
	diff = np.abs(dy1/dp)
	diff_v = np.abs(dyv/dp)
	if (diff < dy1dp_max_light) and (ir11[ptop] > ir11[ptop -1]):
		dy1dp_max_light = diff
		max_light_val_ir = ptop
	if (diff_v < dyvdp_max_light) and (v1[ptop] > v1[ptop -1]):
		dyvdp_max_light = diff_v
		max_light_val_v = ptop

for ptop in range(351,400):
	dp = 0.01
	dy1 = ir11[ptop] - ir11[ptop - 1]
	dyv = v1[ptop] - v1[ptop - 2]
	diff = np.abs(dy1/dp)
	diff_v = np.abs(dyv/dp)

	if (diff < dy1dp_min_light) and (ir11[ptop] > ir11[ptop -1]):
		dy1dp_min_light = diff
		min_light_val_ir = ptop
	if (diff_v < dyvdp_min_light) and (v1[ptop] > v1[ptop -1]):
		dyvdp_min_light = diff_v
		min_light_val_v = ptop

for ptop in range(301, 400):
	dp = 0.01
	dyv = v1[ptop] - v1[ptop - 10]
	diff_v = np.abs(dyv/dp)
	if (diff_v < dyvdp_min_light) and (v1[ptop] > v1[ptop -1]):
		dyvdp_min_light = diff_v
		min_light_val_v = ptop

for ptop in range(301, 400):
	dp = 0.01
	dyv = v1[ptop] - v1[ptop - 5]
	diff_v = np.abs(dyv/dp)
	if (diff_v < dyvdp_min_light) and (v1[ptop] > v1[ptop -1]):
		dyvdp_max_light = diff_v
		max_light_val_v = ptop



#print "max light ", max_light_val, dy1dp_max_light, ir11[max_light_val], ir1x[max_light_val]
#print "min light ", min_light_val, dy1dp_min_light, ir11[min_light_val], ir1x[min_light_val]

if ir11[min_light_val_ir] > ir11[max_light_val_ir]:
	min_phase_ir = ir1x[min_light_val_ir] - np.floor(ir1x[min_light_val_ir])
else:
	min_phase_ir = ir1x[max_light_val_ir] - np.floor(ir1x[max_light_val_ir])
	
if v1[min_light_val_v] > v1[max_light_val_v]:
	min_phase_v = vx[min_light_val_v] - np.floor(vx[min_light_val_v])
else:
	min_phase_v = vx[max_light_val_v] - np.floor(vx[max_light_val_v])
	

ir1x = ir1x - min_phase_ir
vx = vx - min_phase_v
xphaseir1 = xphaseir1 - min_phase_ir
xphasev = xphasev - min_phase_v


mp.clf()

#fig = plt.figure()
#ax1 = fig.add_subplot(111)
#mp.figure(figsize=(16.0,10.0))

cepname = re.sub('HV00', 'HV ', cepname)
cepname = re.sub('HV0', 'HV', cepname)
cepname = re.sub('HV','HV ', cepname)

outname = re.sub(' ','',cepname) 

ax1 = mp.subplot(111)
ax1.axis([0.5,3.0,(avev + 1),(avev-1)])
titlestring = cepname + ', P = ' + str(period) + ' days'
#print titlestring
mp.suptitle(titlestring, fontsize=20)

ax1.set_ylabel('Magnitude')
ax1.set_xlabel('Phase $\phi$')


## Fitting and plotting for each band
		
ax1.plot(vx,v1,'k-')
ax1.plot(xphasev,yv,mfc='ForestGreen', marker='o',ls='None', label='$V$')

ir_lab = '$[3.6] +$ {0:.1f}'.format(offset)

#ax1.plot(ir1x+shift,off_ir11,'k-')
#ax1.plot(xphaseir1+shift,off_yir1,color='MediumVioletRed',marker='o',ls='None', label=ir_lab)

#handles, labels = ax1.get_legend_handles_labels() 
#ax1.legend(handles[::-1],labels[::-1],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., numpoints=1)
#ax1.legend(handles[::-1],labels[::-1],loc=4, numpoints=1,prop={'size':10})


plotname = outname+'_v.pdf'
mp.savefig(plotname, transparent='True')

avsout.close()
mp.show()
#fitout.close()
	
	







															
		

		
	


