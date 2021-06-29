# -------------------------------------------------------------
# Calibration procedure to derive the shape of the AOTF filter
# Developed for the ExoMars/TGO/NOMAD instrument
# Villanueva, Liuzzi, Aoki - NASA/GSFC - June 2021
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
import os, sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# AOTF shape parameters
aotfwc  = [-2.85452723e-07,  1.66652129e-03,  1.83411690e+01] # Sinc width [cm-1 from AOTF frequency cm-1]
aotfsc  = [ 2.19386777e-06, -1.32919656e-02,  2.18425092e+01] # sidelobes factor [scaler from AOTF frequency cm-1]
aotfac  = [-3.35834373e-10, -6.10622773e-05,  1.62642005e+00] # Asymmetry factor [scaler from AOTF frequency cm-1]
aotfoc  = [ 1.55368929e-07, -8.21924887e-04,  1.28833327e+00] # Offset [coefficients for AOTF frequency cm-1]

# Calibration coefficients (Liuzzi+2019 with updates in Aug/2019)
cfaotf  = np.array([1.34082e-7, 0.1497089, 305.0604])        # Frequency of AOTF [cm-1 from kHz]
cfpixel = np.array([1.75128E-08, 5.55953E-04, 2.24734E+01])  # Blaze free-spectral-range (FSR) [cm-1 from pixel]
tcoeff  = np.array([-0.736363, -6.363908])                   # Blaze frequency shift due to temperature [pixel from Celsius]
aotfts  = -6.5278e-5                                         # AOTF frequency shift due to temperature [relative cm-1 from Celsius]
blazep  = [0.22,150.8]                                       # Blaze pixel location with order [pixel from order]
norder  = 6                                                  # Number of +/- orders to be considered in order-addition
npix    = 30                                                 # Number of +/- pixels to be analyzed
forder  = 3                                                  # Polynomial order used to fit the baseline

# Define auxliary functions
def sinc(dx,width,lobe,asym,offset):
	sinc = (width*np.sin(np.pi*dx/width)/(np.pi*dx))**2.0
	ind = (abs(dx)>width).nonzero()[0]
	if len(ind)>0: sinc[ind] = sinc[ind]*lobe
	ind = (dx<=-width).nonzero()[0]
	if len(ind)>0: sinc[ind] = sinc[ind]*asym
	sinc += offset/(2.0*norder + 1.0)
	return sinc
#End sinc-function

def model(solar, xmin, xmax, norder, aotff, aotfw, aotfs, aotfa, aotfo, blaze0, blazew):
	global lint
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
		aotf = sinc(solar[j1:j2,0]-aotff, aotfw, aotfs, aotfa, aotfo)
		if (i+ord)==lorder: spec[:,2] += ss*aotf; spec[:,4] += aotf
		else: spec[:,3] += ss*aotf; spec[:,5] += aotf
		spec[:,1] += ss*aotf
	#Endfor
	#for i in [1,3,5]: spec[:,i] += spec[:,i] + aotfo

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

def maotf(x, *p):
	df,wdt,sdl,asm,off = p
	i1 = lpix - npix; i2 = lpix + npix
	order = round(line/(np.polyval(cfpixel,160.0)))
	blazep0 = round(np.polyval(blazep,order))     # Center location of the blaze in pixels
	blazew = np.polyval(cfpixel,blazep0)      	  # Blaze width [cm-1]
	dpix = np.polyval(cfpixel,range(i1,i2))       # Dispersion parameters per pixel
	spec0 = np.zeros(i2-i1); spec = np.zeros(i2-i1); mm = []

	for i in range(len(x)):
		freq = x[i] + line + df
		order = round(freq/blazew)
		spec0[:] = 0.0; spec[:] = 0.0;
		for j in range(-norder,norder+1):
			dx = dpix*(order + j) - freq
			aotf = sinc(dx, wdt, sdl, asm, off)
			if (j+order)==lorder: spec0 += aotf
			spec += aotf
		#Endfor
		#spec += off
		mm.append(np.mean(spec0)/np.mean(spec))
	#Endfor

	return mm
#End full 2D AOTF model

# Plot scheme
if False:
	solar = np.genfromtxt('aotf/solar_3787.txt'); solar = np.flip(solar, 0); solar[:,1] /= 1.055
	fig, ax = plt.subplots(1,2,figsize=[12,5])
	ax[0].plot(solar[:,0],solar[:,1])
	ax[0].plot(solar[:,0],(solar[:,1]+0.6)/(1.0+0.6),color='red')
	ax[0].plot([3785,3795],[1,1],linestyle=':',color='black')
	ax[0].plot([3785,3795],[0,0],linestyle=':',color='black')
	ax[0].set_xlim([3785,3795])
	ax[0].set_ylim([-0.05,1.1])
	ax[1].plot(solar[:,0],(solar[:,1]+0.6)/(1.0+0.6),color='red')
	ax[1].plot([3785,3795],[1,1],linestyle=':',color='black')
	ax[1].plot([3785,3795],[0,0],linestyle=':',color='black')
	ax[1].set_xlim([3785,3795])
	ax[1].set_ylim([-0.05,1.1])
	ax[1].fill([3785,3795,3795,3785],[0.0,0.0,0.6/1.6,0.6/1.6],'red')
	plt.tight_layout()
	plt.savefig('aotf_offset.png')
	plt.show()

	dx = np.arange(-140,140,0.1)
	plt.figure(figsize=[10,6])
	plt.plot(dx,sinc(dx, 20, 1.0, 1.0))
	plt.plot(dx,sinc(dx, 20, 3.0, 1.0))
	plt.plot(dx,sinc(dx, 20, 3.0, 2.0))
	plt.plot([-100,100],[0,0], color='black',linestyle=':')
	plt.plot([-100,100],[1,1], color='black',linestyle=':')
	plt.xlabel('AOTF frequency - center [cm-1]')
	plt.ylabel('AOTF relative intensity')
	plt.xlim([-100,100])
	plt.tight_layout()
	plt.savefig('aotf.png')
	plt.show()
#Endif

# Iterate acros lines
for il in range(14):
	#if il<4: continue
	# Line to characterize
	fits = [True,True,True,True]
	if il==0:  line = 2703.8; fits[2] = False
	if il==1:  line = 2733.3; fits[2] = False
	if il==2:  line = 2837.8
	if il==3:  line = 2927.1
	if il==4:  line = 2942.5; fits[0] = False; fits[2] = False
	if il==5:  line = 3172.9
	if il==6:  line = 3289.6
	if il==7:  line = 3414.5; continue
	if il==8:  line = 3650.9; fits[0] = False; fits[2] = False
	if il==9:  line = 3750.1; fits[2] = False
	if il==10: line = 3755.8; fits[2] = False
	if il==11: line = 3787.9
	if il==12: line = 4276.1
	if il==13: line = 4383.5

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
	lorder = round(line/(np.polyval(cfpixel,160.0)))
	xdat  = np.polyval(cfpixel,range(320))*lorder
	lpix = int(np.interp(line, xdat, range(320)))
	lfile = 'aotf/aotf_%d.txt' % line
	if not os.path.isdir('aotf'): os.system('mkdir aotf')
	print(line, lpix)

	# List of observations
	# https://nomad.aeronomie.be/index.php/observations/calibration-observations
	inst = 'SO'; files = os.listdir(inst); files.sort(); xpeaks=[]; ypeaks=[]; mpeaks=[]; sets=[]; scans=[]; set=-1; scan=0
	for file in files:
		if file[-2:]!='h5': continue
		if os.path.exists(lfile): continue
		#if file!='20190307_004053_0p2a_SO_1_C.h5': continue
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
		temp  = np.array(list(hdf5f['/Housekeeping/SENSOR_1_TEMPERATURE_'+inst])); mtemp = np.median(temp[10:30])
		tempa = np.array(list(hdf5f['/Housekeeping/AOTF_TEMP_'+inst])); mtempa = np.median(tempa[10:30])
		data  = np.array(list(hdf5f['/Science/Y']))
		data  = np.sum(data,axis=1) # Add all bins
		hdf5f.close()
		print(file)

		lfreq=0; lshift=0.0; shifts=[]
		for i in range(npts):
			# Define the frequency solution
			freq  = np.polyval(cfaotf,aotf[i])
			freq += aotfts*mtempa*freq;
			if freq<line-140 or freq>line+140: continue
			#if freq<lfreq and lfreq>0: continue
			if freq<lfreq or lfreq==0: set += 1; scan = 0
			order = round(freq/(np.polyval(cfpixel,160.0)))
			ipix  = range(320)
			xdat  = np.polyval(cfpixel,ipix)*order
			dpix = np.polyval(tcoeff,mtemp)
			xdat += dpix*(xdat[-1]-xdat[0])/320.0

			# Define the normalized intensity (and error)
			blazep0 = round(np.polyval(blazep,order)) # Center location of the blaze in pixels
			blaze0 = xdat[blazep0]                    # Blaze center frequency [cm-1]
			blazew = np.polyval(cfpixel,blazep0)      # Blaze width [cm-1]
			dx = xdat - blaze0; dx[blazep0] = 1e-6

			dpix = int(np.round(dpix))
			i1 = lpix - npix - dpix; i2 = lpix + npix - dpix
			xpix = ipix[i1:i2]
			xdat = xdat[i1:i2]
			blaze = (blazew*np.sin(np.pi*dx/blazew)/(np.pi*dx))**2
			ydat = data[i,i1:i2]/blaze[i1:i2]
			ydat = ydat/np.mean(ydat)

			# Compute the model
			smodel = model(solar, xdat[0]-0.2,xdat[-1]+0.2, norder, freq, aotfw, aotfs, aotfa, aotfo, blaze0, blazew)

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
			cf,vm = curve_fit(contrib, xdat, ydat, p0=[1], bounds=[0.0,50])

			if False:
				mdat = np.interp(xdat, smodel[:,0], smodel[:,1])
				gain = np.polyval(np.polyfit(xpix, ydat/mdat, forder), xpix)
				ot = contrib(xdat, *cf)
				plt.plot(ydat); plt.plot(ot); plt.plot(mdat*gain)
				plt.title('%f' % (freq-line))
				plt.savefig('anim/anim-%03d.png' % scan)
				plt.show(block=False); plt.pause(0.1); plt.cla()
			#Endif

			# Store results
			xpeaks.append(freq-line); lfreq = freq
			ypeaks.append(np.mean(cf[0]*smodel[:,4]/(cf[0]*smodel[:,4] + smodel[:,5])))
			mpeaks.append(np.mean(smodel[:,4]/(smodel[:,4] + smodel[:,5])))
			sets.append(set); scans.append(scan); scan += 1
		#Endfor
	#Endfor
	if len(xpeaks)>0:
		data = np.zeros([len(xpeaks),5])
		data[:,0] = sets
		data[:,1] = scans
		data[:,2] = xpeaks
		data[:,3] = ypeaks
		data[:,4] = mpeaks
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
	fmin0 = np.min(data[0,:,0]); fmax0 = np.max(data[0,:,0]);
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
	dc1=-3.0; dc2=3.0
	dw1=18.0; dw2=25.0
	ds1=0.20; ds2=10.0
	da1=0.20; da2=10.0
	do1=0.00; do2=1.00
	if not fits[0]: dw1=aotfw-1e-6; dw2=aotfw+1e-6
	if not fits[1]: ds1=aotfs-1e-6; ds2=aotfs+1e-6
	if not fits[2]: da1=aotfa-1e-6; da2=aotfa+1e-6
	if not fits[3]: do1=aotfo-1e-6; do2=aotfo+1e-6
	cf,vm = curve_fit(maotf, dt[:,2], dt[:,3], p0=[0.0,aotfw,aotfs,aotfa,aotfo], bounds=[[dc1,dw1,ds1,da1,do1],[dc2,dw2,ds2,da2,do2]])
	#cf = [0.0,aotfw,aotfs,aotfa,aotfo]
	#cf = [-0.74,20.12,5.24,1.40,1.26]
	print(line, cf[0],cf[1],cf[2],cf[3],cf[4])
	fl = open('aotf/fit_%d.txt' % line,'w'); fl.write('%.1f %.3f %.3f %.3f %.4f %d\n' % (line,cf[1],cf[2],cf[3],cf[4],lpix)); fl.close()

	# Plot data
	plt.cla()
	for i in range(nsets): plt.plot(-data[i,0:nscans[i],0]-cf[0],data[i,0:nscans[i],1],linewidth=0.5)
	cf[0] = 0.0; mm = maotf(dt[:,2], *cf)
	plt.plot(-dt[:,2], mm, color='black', linewidth=2.0)
	ms = sinc(dt[:,2]+cf[0],cf[1],cf[2],cf[3],cf[4])
	ms *= np.max(mm)
	plt.plot( dt[:,2], ms, color='blue', linestyle=':')
	plt.xlim([-140,140])
	plt.title('Line: %.1f Params: %.3f %.3f %.3f %.4f' % (line,cf[1],cf[2],cf[3],cf[4]))
	plt.tight_layout()
	plt.savefig('aotf/aotf_%d.png' % line)
	#plt.show(); exit()
	plt.show(block=False); plt.pause(0.1)
#Endfor
plt.show(block=False); plt.pause(5.0)
