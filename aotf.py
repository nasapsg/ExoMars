# -------------------------------------------------------------
# Calibration procedure to derive the shape of the AOTF filter
# Developed for the ExoMars/TGO/NOMAD instrument
# Villanueva, Liuzzi, Aoki - NASA/GSFC - Sep 2021
# -------------------------------------------------------------
# To run type 'python aotf.py'. First the code will iterate
# across every selected line and search for mini-scans that sample
# that line. It will then extract the solar line intensity by fitting
# a gaussian, after removing a baseline around it. This information
# at each AOTF frequency is then collected and saved in aotf_LINE.txt.
# The second part of the code reads this file, and fits an
# asymmetric single-sinc AOTF function to this, while also modeling
# contamination signatures from nearby orders.
# -------------------------------------------------------------
import os, sys, copy
import h5py
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import mpfit
#from lmfit import Parameters, minimize

# AOTF shape parameters
aotfwc  = [-1.66406991e-07,  7.47648684e-04,  2.01730360e+01] # Sinc width [cm-1 from AOTF frequency cm-1]
aotfsc  = [ 8.10749274e-07, -3.30238496e-03,  4.08845247e+00] # sidelobes factor [scaler from AOTF frequency cm-1]
aotfac  = [-1.54536176e-07,  1.29003715e-03, -1.24925395e+00] # Asymmetry factor [scaler from AOTF frequency cm-1]
aotfoc  = [            0.0,             0.0,             0.0] # Offset [coefficients for AOTF frequency cm-1]
aotfgc  = [ 1.49266526e-07, -9.63798656e-04,  1.60097815e+00] # Gaussian peak intensity [coefficients for AOTF frequency cm-1]

# Calibration coefficients
cfaotf  = [1.34082e-7, 0.1497089, 305.0604]                   # Frequency of AOTF [cm-1 from kHz]
cfpixel = [1.75128E-08, 5.55953E-04, 2.24734E+01]             # Blaze free-spectral-range (FSR) [cm-1 from pixel]
ncoeff  = [-2.44383699e-07, -2.30708836e-05, -1.90001923e-04] # Relative frequency shift coefficients [shift/frequency from Celsius]
aotfts  = -6.5278e-5                                          # AOTF frequency shift due to temperature [relative cm-1 from Celsius]
blazep  = [-1.00162255e-11, -7.20616355e-09, 9.79270239e-06, 2.25863468e+01] # Dependence of blazew from AOTF frequency
norder  = 6                                                   # Number of +/- orders to be considered in order-addition
npix    = 30                                                  # Number of +/- pixels to be analyzed
forder  = 3                                                   # Polynomial order used to fit the baseline

# Define auxliary functions
def sinc(dx,width,lobe,asym,offset,gauss):
	sinc = (width*np.sin(np.pi*dx/width)/(np.pi*dx))**2.0
	ind = (abs(dx)>width).nonzero()[0]
	if len(ind)>0: sinc[ind] = sinc[ind]*lobe
	ind = (dx<=-width).nonzero()[0]
	if len(ind)>0: sinc[ind] = sinc[ind]*asym
	sinc += offset/(2.0*norder + 1.0)
	sigma = 50.0
	sinc += gauss*np.exp(-0.5*(dx/sigma)**2.0)
	return sinc
#End sinc-function

def model(solar, xmin, xmax, norder, aotff, aotfw, aotfs, aotfa, aotfo, aotfg, blaze0, blazew):
	i1 = 0
	while solar[i1,0]<xmin: i1+=1
	i2 = i1
	while solar[i2,0]<xmax: i2+=1
	RP = solar[0,0]/(solar[1,0]-solar[0,0])
	spec = np.zeros([i2-i1,6]); spec[:,0] = solar[i1:i2,0]
	ord = round(blaze0/blazew)
	for i in range(-norder,norder+1):
		nsh = int(np.round(np.log((blaze0 + i*blazew)/blaze0)/np.log((1.0+(1.0/RP)))))
		j1 = i1 + nsh; j2 = i2 + nsh
		if j1<0 or j2>=solar.shape[0]: continue
		ss = solar[j1:j2,1]
		aotf = sinc(solar[j1:j2,0]-aotff, aotfw, aotfs, aotfa, aotfo, aotfg)
		if (i+ord)==lorder: spec[:,2] += ss*aotf; spec[:,4] += aotf
		else: spec[:,3] += ss*aotf; spec[:,5] += aotf
		spec[:,1] += ss*aotf
	#Endfor

	# Add ghost ILS
	dfp = np.polyval([-2.4333526e-06,0.0018259633,-0.031606901],xpix)*(xmin+xmax)/(2.0*3700.0)
	dfh = np.interp(spec[:,0], xdat, dfp)
	for i in [1,2,3,4,5]:
		ghost = np.interp(spec[:,0]+dfh,spec[:,0],spec[:,i])
		spec[:,i] = (spec[:,i] + ghost*0.25)/1.25
	#Endfor
	return spec
#Enddef

def contrib(x, *p):
	mdat = oo + o0*p
	gain = np.polyval(np.polyfit(xpix, ydat/mdat, forder), xpix)
	return mdat*gain
#Enddef

def maotf(p, fjac=None, x=None, y=None, err=None):
	global scl; scl=1.0
	df,wdt,sdl,asm,off,gint = p
	blazew =  np.polyval(blazep,line-3700.0)
	order = round(line/blazew)
	dxmin = -(norder+1)*blazew; dxmax = (norder+1)*blazew
	xm = np.arange(dxmin, dxmax, 0.1)
	yf = sinc(xm, wdt, sdl, asm, off, gint)
	ym = np.zeros(len(xm))
	y0 = np.zeros(len(xm))
	for i in range(len(xm)):
		fp = xm[i] + line + df
		op = round(fp/blazew)
		d0 = fp - line*op/lorder
		for j in range(-norder,norder+1):
			dx = fp*(op+j)/(op) - fp - d0
			if dx<dxmin or dx>dxmax: continue
			val = np.interp(dx, xm, yf)
			ym[i] += val
			if (op+j)==order: y0[i] += val
		#Endfor
	#Endfor
	y0 /= ym
	mm = np.interp(x, xm, y0)
	#scl = np.sum(y)/np.sum(mm)
	#mm *= scl
	mm += (np.sum(y) - np.sum(mm))/len(mm)
	#chsq = np.sum(((y-mm)/err)**2.0)
	#print(chsq, p)
	return([0, (y-mm)/err])
#End full 2D AOTF model

# Iterate acros lines
for il in range(16):
	if il!=2: continue
	# Line to characterize
	fits = [True,True,True,False,True]; dfa=0.0; #0:width, 1:sidelobe, 2:asym, 3:offset; 4:gaussian
	if il==0:  line = 2703.8; dfa= 2.0; fits[2] = False
	if il==1:  line = 2733.3; dfa= 1.1; fits[2] = False
	if il==2:  line = 2837.8; dfa= 1.4; fits[0] = False
	if il==3:  line = 2927.1; dfa= 0.7;
	if il==4:  line = 2942.5; dfa= 0.5; fits[2] = False
	if il==5:  line = 3172.9; dfa= 0.2
	if il==6:  line = 3289.6; dfa= 0.4
	if il==7:  line = 3414.5; dfa=-0.4; continue # Strange narrow shape
	if il==8:  line = 3650.9; dfa= 0.8; continue # Strange narrow shape
	if il==9:  line = 3750.1; dfa= 1.2
	if il==10: line = 3755.8; dfa= 0.0; continue # Strange broad shape
	if il==11: line = 3787.9; dfa= 1.1
	if il==12: line = 4276.1; dfa=-0.7; continue
	if il==13: line = 4383.5; dfa=-0.1
	if il==14: line = 4448.5; dfa=-0.8;
	if il==15: line = 4088.1; dfa= 0.0; continue

	# Get Solar spectrum
	if not os.path.isdir('aotf'): os.system('mkdir aotf')
	file = 'aotf/solar_%d.txt' % line
	if not os.path.exists(file):
		#server = 'http://localhost' # URL of PSG server (local)
		server = 'https://psg.gsfc.nasa.gov' # URL of PSG server
		fl = open("psg_solar.txt", "r"); cfg = fl.readlines(); fl.close()
		cfg[0] = '<GENERATOR-RANGE1>%.1f\n' % (line - 150 - norder*23.0)
		cfg[1] = '<GENERATOR-RANGE2>%.1f\n' % (line + 150 + norder*23.0)
		fl = open("config.txt", "w")
		for lnn in cfg: fl.write(lnn)
		fl.close()
		os.system('curl -s --data-urlencode file@config.txt %s/api.php > %s' % (server,file))
	#Endif
	solar = np.genfromtxt(file); solar = np.flip(solar, 0)

	# Define initial AOTF parameters and the line
	aotfw = np.polyval(aotfwc,line)
	aotfs = np.polyval(aotfsc,line)
	aotfa = np.polyval(aotfac,line)
	aotfo = np.polyval(aotfoc,line)
	aotfg = np.polyval(aotfgc,line)
	blazew =  np.polyval(blazep,line-3700.0)
	lorder = round(line/blazew)
	xdat   = np.polyval(cfpixel,range(320))*lorder
	lpix   = int(np.interp(line, xdat, range(320)))
	lfile  = 'aotf/aotf_%d.txt' % line
	if not os.path.isdir('aotf'): os.system('mkdir aotf')

	# List of observations
	# https://nomad.aeronomie.be/index.php/observations/calibration-observations
	inst = 'SO'; files = os.listdir(inst); files.sort(); xpeaks=[]; ypeaks=[]; mpeaks=[]; sets=[]; scans=[]; shifts=[]; temps=[]; shifts0=[]; set=-1; scan=0
	for file in files:
		if file[-2:]!='h5': continue
		#if file=='20180716_000706_0p2a_SO_1_C.h5': continue
		#if file!='20190819_001536_0p2a_SO_1_C.h5': continue
		if os.path.exists(lfile): continue
		year = int(file[:4])
		hdf5f = h5py.File('%s/%s' % (inst,file), "r")
		aotf = np.array(list(hdf5f['/Channel/AOTFFrequency']))
		freq = np.array([np.polyval(cfaotf,x) for x in aotf])
		ind = (freq>2000).nonzero()[0]
		fmin = np.min(freq[ind])
		fmax = np.max(freq)
		npts = len(aotf)
		if fmin>line-10 or fmax<line+10 or (aotf[-1]-aotf[-2])>10: continue

		# Read data
		data  = np.array(list(hdf5f['/Science/Y']))
		data  = np.sum(data,axis=1) # Add all bins

		# Define temperature
		#temp  = np.array(list(hdf5f['/Housekeeping/SENSOR_1_TEMPERATURE_'+inst]))
		#tmeds = medfilt(temp[-npts:],5)
		#tempa = np.array(list(hdf5f['/Housekeeping/AOTF_TEMP_'+inst])); mtempa = np.median(tempa[10:30])
		temp = np.array(list(hdf5f['/Channel/InterpolatedTemperature']))
		print(file)

		lfreq=0; lshift=0.0;
		for i in range(npts):
			# Define the frequency solution
			freq  = np.polyval(cfaotf,aotf[i]) + dfa
			freq += aotfts*temp[i]*freq;
			if freq<line-140 or freq>line+140: continue
			#if freq<line-1 or freq>line+1: continue
			if freq<lfreq or lfreq==0: set += 1; scan = 0
			blazew =  np.polyval(blazep,freq-3700.0)         # FSR (Free Spectral Range), blaze width [cm-1]
			blazew += blazew*np.polyval(ncoeff,temp[i])     # FSR, corrected for temperature
			order = round(freq/blazew)                       # Grating order
			ipix  = range(320)
			xdat  = np.polyval(cfpixel,ipix)*order
			sh0   = xdat[lpix]*np.polyval(ncoeff, temp[i])
			xdat += xdat*np.polyval(ncoeff, temp[i])

			# Define the normalized intensity (and error)
			blaze0 = order*blazew
			dx = xdat - blaze0
			ind = ((dx==0.0).nonzero())[0]
			if len(ind)>0: dx[ind]=1e-6
			dpix = int(np.round(sh0/(xdat[lpix]-xdat[lpix-1])))
			i1 = lpix - npix - dpix; i2 = lpix + npix - dpix
			xpix = ipix[i1:i2]
			xdat = xdat[i1:i2]
			blaze = (blazew*np.sin(np.pi*dx/blazew)/(np.pi*dx))**2
			data[i,269] = (data[i,268]+data[i,270])*0.5
			data[i,256] = (data[i,255]+data[i,257])*0.5
			ydat = data[i,i1:i2]/blaze[i1:i2]
			ydat = ydat/np.mean(ydat)

			# Compute the model
			smodel = model(solar, xdat[0]-0.2,xdat[-1]+0.2, norder, freq, aotfw, aotfs, aotfa, aotfo, aotfg, blaze0, blazew)

			# Search for small frequency corrections
			dsh = (xdat[1]-xdat[0])*0.2; corrs=[]
			for nsh in range(-20,20+1):
				shift = dsh*nsh
				mdat = np.interp(xdat + shift, smodel[:,0], smodel[:,1])
				gain = np.polyval(np.polyfit(xpix, ydat/mdat, forder), xpix)
				corr = np.corrcoef(ydat/gain, mdat)[1,0]; corrs.append(corr)
			#Endfor
			shift = dsh*(np.argmax(corrs)-20);
			xdat += shift

			# Determine the contribution from the center order
			o0 = np.interp(xdat, smodel[:,0], smodel[:,2])
			oo = np.interp(xdat, smodel[:,0], smodel[:,3])
			cf,vm = curve_fit(contrib, xdat, ydat, p0=[1], bounds=[0.02,50.0], maxfev=5000)

			if False:
				mdat = np.interp(xdat, smodel[:,0], smodel[:,1])
				gain = np.polyval(np.polyfit(xpix, ydat/mdat, forder), xpix)
				ot = contrib(xdat, *cf)
				plt.plot(ydat); plt.plot(ot,label='Fitted'); plt.plot(mdat*gain,label='Orig')
				#plt.plot(blaze/np.max(blaze)); plt.plot(data[i,:]/np.max(data[i,:]))
				plt.title('%f %f' % (freq-line, blazew))
				#plt.savefig('anim/anim-%03d.png' % scan)
				plt.legend(loc='upper left')
				plt.show(block=False); plt.pause(0.01); plt.cla()
			#Endif

			# Store results
			xpeaks.append(freq-line); lfreq = freq
			ypeaks.append(np.mean(cf[0]*smodel[:,4]/(cf[0]*smodel[:,4] + smodel[:,5])))
			mpeaks.append(np.mean(smodel[:,4]/(smodel[:,4] + smodel[:,5])))
			shifts.append(shift + sh0); shifts0.append(sh0); temps.append(temp[i])
			sets.append(set); scans.append(scan); scan += 1
		#Endfor
	#Endfor
	if len(xpeaks)>0:
		data = np.zeros([len(xpeaks),8])
		data[:,0] = sets
		data[:,1] = scans
		data[:,2] = xpeaks
		data[:,3] = ypeaks
		data[:,4] = mpeaks
		data[:,5] = shifts
		data[:,6] = temps
		data[:,7] = shifts0
		np.savetxt(lfile, data)
	#Endif

	# Read data
	if not os.path.exists(lfile): exit()
	dt = np.genfromtxt(lfile)
	nsets = int(np.max(dt[:,0])+1)
	mscans= int(np.max(dt[:,1])+1); nscans = []
	data = np.zeros([nsets, mscans, 3])
	for i in range(dt.shape[0]):
		data[int(dt[i,0]),int(dt[i,1]),:] = dt[i,2:5]
		if dt[i,0]>0 and int(dt[i,1])==0: nscans.append(nn)
		nn = int(dt[i,1])
	#Endfor
	nscans.append(nn)

	# Correct for small AOTF center deviations between scans
	fmin0 = np.min(data[0,:,0]); fmax0 = np.max(data[0,:,0]); scl=1.0
	for i in range(1,nsets):
		corrs=[]; dsh=0.2
		for nsh in range(-10,10+1):
			shift = nsh*dsh; i1=0; i2=0
			fmin = np.max([fmin0, np.min(data[i,0:nscans[i],0])+shift])
			fmax = np.min([fmax0, np.max(data[i,0:nscans[i],0])+shift])
			if (fmax-fmin)<40: continue
			while data[0,i1,0]<fmin: i1+=1; i2=i1
			while data[0,i2,0]<fmax: i2+=1
			tt = np.interp(data[0,i1:i2,0], data[i,0:nscans[i],0]+shift, data[i,0:nscans[i],1])
			corr = np.corrcoef(tt, data[0,i1:i2,1])[1,0]; corrs.append(corr)
		#Endfor
		if len(corrs)>0:
			shift = dsh*(np.argmax(corrs)-10)
			ind = (dt[:,0]==i).nonzero()[0]
			dt[ind,0] += shift
			data[i,:,0] += shift
		#Endif
	#Endfor
	dt = dt[np.argsort(dt[:,2]),:]

	# Fit the AOTF parameters
	parbase={'value':0., 'fixed':0, 'limited':[1,1], 'limits':[0.,0.], 'step':0.01}; parinfo=[]; p0=[]
	for i in range(6): parinfo.append(copy.deepcopy(parbase))
	parinfo[0]['limits'][0] = -3.0; parinfo[0]['limits'][1] =  3.0; parinfo[0]['value'] = 0.0
	parinfo[1]['limits'][0] = 18.0; parinfo[1]['limits'][1] = 25.0; parinfo[1]['value'] = aotfw
	parinfo[2]['limits'][0] =  0.2; parinfo[2]['limits'][1] = 10.0; parinfo[2]['value'] = aotfs
	parinfo[3]['limits'][0] =  0.2; parinfo[3]['limits'][1] = 10.0; parinfo[3]['value'] = aotfa
	parinfo[4]['limits'][0] =  0.0; parinfo[4]['limits'][1] =  1.0; parinfo[4]['value'] = aotfo
	parinfo[5]['limits'][0] =  0.0; parinfo[5]['limits'][1] =  1.0; parinfo[5]['value'] = aotfg
	for i in range(5): parinfo[i+1]['fixed'] = 1 - fits[i]
	for i in range(6): p0.append(parinfo[i]['value'])
	x = dt[:,2]; y = dt[:,3]; err=(y*0.0+0.1)
	fa = {'x':x, 'y':y, 'err':err}
	m = mpfit.mpfit(maotf, p0, parinfo=parinfo, functkw=fa, xtol=1.e-2, maxiter=20, quiet=True)
	cf = m.params;
	print(line, cf[0]+dfa,cf[1],cf[2],cf[3],cf[4],cf[5], scl)
	fl = open('aotf/fit_%d.txt' % line,'w'); fl.write('%.1f %.3f %.3f %.3f %.4f %.4f %.4f %d\n' % (line,cf[1],cf[2],cf[3],cf[4],cf[5],scl,lpix)); fl.close()

	# Plot data
	plt.cla()
	for i in range(nsets): plt.plot(data[i,0:nscans[i],0]+cf[0],data[i,0:nscans[i],1],linewidth=0.5)
	cf[0] = 0.0
	fit = maotf(cf, x=x, y=y, err=err)
	fit = y - fit[1]*err
	plt.plot(x, fit, color='black', linewidth=2.0)
	plt.xlim([-140,140])
	plt.title('Line: %.1f Params: %.3f %.3f %.3f %.4f %.4f' % (line,cf[1],cf[2],cf[3],cf[4],cf[5]))
	plt.tight_layout()
	plt.savefig('aotf/aotf_%d.png' % line)
	plt.show(block=False); plt.pause(0.1)
	#plt.show(); exit()
#Endfor
plt.show(block=False); plt.pause(5.0)
