# -------------------------------------------------------------
# Calibration procedure to derive the shape of the AOTF filter
# Developed for the ExoMars/TGO/NOMAD instrument
# Villanueva, Liuzzi - NASA/GSFC - April 2021
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

fitwd = False
fitsl = False
fitaf = False

# AOTF shape parameters
aotfwc  = [1.11085173e-06, -8.88538288e-03,  3.83437870e+01] # Sinc width [cm-1 from AOTF frequency cm-1]
aotfsc  = [2.87490586e-06, -1.65141511e-02,  2.49266314e+01] # sidelobes factor [scaler from AOTF frequency cm-1]
aotfaf  = [-5.47912085e-07, 3.60576934e-03, -4.99837334e+00] # Asymmetry factor [scaler from AOTF frequency cm-1]

# Calibration coefficients (Liuzzi+2019 with updates in Aug/2019)
cfaotf  = np.array([1.34082e-7, 0.1497089, 305.0604])        # Frequency of AOTF [cm-1 from kHz]
cfpixel = np.array([1.75128E-08, 5.55953E-04, 2.24734E+01])  # Blaze free-spectral-range (FSR) [cm-1 from pixel]
tcoeff  = np.array([-0.736363, -6.363908])                   # Blaze frequency shift due to temperature [pixel from Celsius]
aotfts  = -6.5278e-5                                         # AOTF frequency shift due to temperature [relative cm-1 from Celsius]
blazep  = [0.22,150.8]                                       # Blaze pixel location with order [pixel from order]

# Define auxliary functions
def gauss(x, *p):
	A, mu, b0,b1,b2 = p
	return A*np.exp(-(x-mu)**2/(2.0*sigma**2)) + b0+b1*x+b2*x*x
#End gauss model definition
def sinc(dx,amp,width,lobe,asym):
	sinc = amp*(width*np.sin(np.pi*dx/width)/(np.pi*dx))**2
	ind = (abs(dx)>width).nonzero()[0]
	if len(ind)>0: sinc[ind] = sinc[ind]*lobe
	ind = (dx<=-width).nonzero()[0]
	if len(ind)>0: sinc[ind] = sinc[ind]*asym
	return sinc
#End sinc-function
def maotf(x, *p):
	global msinc
	amp,width,pos0,lobe,asym, cf0,cf1,cf2 = p
	dx = (x - pos0)
	model  = sinc(dx,amp,width,lobe,asym); msinc = 1.0*model
	model += np.polyval([cf0,cf1,cf2],dx)
	for i in range(len(cc)):
		mm = np.interp(cc[i]*lblaze-pos0,dx,model)
		mm = ccf[i] - mm
		if mm<0: mm=0.0
		dx = (x - pos0) - cc[i]*lblaze
		model += sinc(dx,mm,width,lobe,asym)
	#Endfor
	return model
#End AOTF model definition

# Iterate acros lines
for il in range(13):
	# Line to characterize
	ob = 2   # Polynomial order of the baseline near the line (+/- 15 pixels)
	cc = []  # Orders where a contamination line is located
	if il==0:  line = 2703.8; sigma=1.6; cc=[1,4]
	if il==1:  line = 2733.3; sigma=1.6; cc=[2]
	if il==2:  line = 2837.8; sigma=1.6; cc=[4,2,-4]
	if il==3:  line = 2927.1; sigma=1.6; cc=[2,-4]
	if il==4:  line = 2942.5; sigma=1.6; cc=[1,3,-5]
	if il==5:  line = 3172.9; sigma=1.6; cc=[4,2,-4]
	if il==6:  line = 3289.6; sigma=1.4; cc=[-6,5,3,-2]
	if il==7:  line = 3414.5; sigma=1.6; cc=[-3,3]
	if il==8:  line = 3650.9; sigma=1.6; cc=[2]
	if il==9:  line = 3750.1; sigma=1.6; cc=[-5]
	if il==10: line = 3755.8; sigma=1.6; cc=[-3,5,-2]
	if il==11: line = 3787.9; sigma=1.6; cc=[1,5]
	if il==12: line = 4383.5; sigma=2.5; ob=1; cc=[-3]

	# Parameters of the line
	lorder = round(line/(np.polyval(cfpixel,160.0)))
	xdat  = np.polyval(cfpixel,range(320))*lorder
	lpix = np.interp(line, xdat, range(320))
	lblaze = xdat[round(lpix)]/lorder
	lfile = 'aotf/aotf_%d.txt' % line
	if not os.path.isdir('aotf'): os.system('mkdir aotf')

	# List of observations
	# https://nomad.aeronomie.be/index.php/observations/calibration-observations
	inst = 'SO'; files = os.listdir(inst); files.sort(); xpeaks=[]; ypeaks=[]
	for file in files:
		if file[-2:]!='h5' or os.path.exists(lfile): continue
		year = int(file[:4])
		hdf5f = h5py.File('%s/%s' % (inst,file), "r")
		aotf = np.array(list(hdf5f['/Channel/AOTFFrequency']))
		freq = np.array([np.polyval(cfaotf,x) for x in aotf])
		order = np.array([round(x/(np.polyval(cfpixel,160.0))) for x in freq]) # Order of the spectrum
		ind = (freq>2000).nonzero()[0]
		fmin = np.min(freq[ind])
		fmax = np.max(freq)
		npts = len(aotf)
		if fmin>line or fmax<line or (aotf[-1]-aotf[-2])>10: continue

		# Read data
		temp  = np.array(list(hdf5f['/Housekeeping/SENSOR_1_TEMPERATURE_'+inst])); mtemp = np.median(temp[10:30])
		tempa = np.array(list(hdf5f['/Housekeeping/AOTF_TEMP_'+inst])); mtempa = np.median(tempa[10:30])
		data  = np.array(list(hdf5f['/Science/Y']))
		data  = np.sum(data,axis=1) # Add all bins
		hdf5f.close()
		print('%.1f %.1f %3d %3d %5d %s' % (np.min(freq[ind]),np.max(freq),np.min(order[ind]),np.max(order),freq.shape[0], file))

		xpeak=[]; ypeak=[]; lfreq=0
		for i in range(npts):
			freq  = np.polyval(cfaotf,aotf[i])
			if freq<line-140 or freq>line+140: continue
			freq += aotfts*mtempa*freq
			order = round(freq/(np.polyval(cfpixel,160.0)))
			ipix  = range(320)
			xdat  = np.polyval(cfpixel,ipix)*order
			dpix = np.polyval(tcoeff,mtemp)
			xdat += dpix*(xdat[-1]-xdat[0])/320.0

			blazep0 = round(np.polyval(blazep,order)) # Center location of the blaze  in pixels
			blaze0 = xdat[blazep0]                    # Blaze center frequency [cm-1]
			blazew = np.polyval(cfpixel,blazep0)      # Blaze width [cm-1]
			dx = xdat - blaze0; dx[blazep0] = 1e-6
			blaze = (blazew*np.sin(np.pi*dx/blazew)/(np.pi*dx))**2
			spec = data[i,:]/(np.max(data[i,:])*blaze)

			ip0 = round(lpix - dpix)
			ip1 = ip0 - 15
			ip2 = ip0 + 15
			w = spec*0.0 + 1.0; w[ip1:ip2+1] = 0.0
			base = np.polyval(np.polyfit(ipix,spec, 6, w=w),ipix)
			spec = 1.0 - spec/base
			xfit = ipix[ip1:ip2] - (lpix-dpix); dcf=0.1
			if ob==1: dcf=1e-10
			cf,vm = curve_fit(gauss, xfit, spec[ip1:ip2], p0=[0.01,0.0,np.mean(spec[ip1:ip2]),0.0,0.0], bounds=[[0.0,-5.0,-0.2,-0.1,-dcf],[0.2,5.0,0.2,0.1,dcf]],maxfev=5000)

			#plt.plot(xfit,spec[ip1:ip2])
			#plt.plot(xfit,gauss(xfit,*cf))
			#plt.title(freq-line)
			#plt.show(block=False); plt.pause(0.1); plt.close(); continue

			if freq<lfreq or i==npts-1:
				if np.min(xpeak)<-1 and np.max(xpeak)>1:
					mmax = np.max(ypeak)
					for j in range(len(xpeak)): xpeaks.append(xpeak[j]); ypeaks.append(ypeak[j]/mmax)
					if year==2018: color='blue'
					if year==2019: color='red'
					if year==2020: color='green'
					plt.plot(xpeak,ypeak/mmax,label=file,color=color)
				#Endif
				xpeak=[]; ypeak=[];
			#Endif
			xpeak.append(freq-line)
			ypeak.append(cf[0])
			lfreq = freq
		#Endfor
	#Endfor

	# Save/plot the results
	if len(xpeaks)>0:
		plt.title(line)
		plt.tight_layout()
		plt.savefig('aotf/aotf_%d.png' % line)
		plt.show()
		data = np.zeros([len(xpeaks),2])
		data[:,0] = xpeaks
		data[:,1] = ypeaks
		np.savetxt(lfile, data)
	#Endif

	# Read the results
	if not os.path.exists(lfile): exit()
	data = np.genfromtxt(lfile)
	data = data[np.argsort(data[:,0]),:]
	ccf = []
	for i in range(len(cc)):
		xp = cc[i]*lblaze
		ind = ((data[:,0]>xp-2)*(data[:,0]<xp+2)).nonzero()[0]
		mm = np.median(data[ind,1])
		ccf.append(mm)
	#Endfor

	wd0 = np.polyval(aotfwc,line); dw1=19.0; dw2=25.0
	sl0 = np.polyval(aotfsc,line); ds1=0.10; ds2=10.0;
	af0 = np.polyval(aotfaf,line); df1=0.10; df2=10.0;
	if not fitwd: dw1=wd0-1e-6; dw2=wd0+1e-6
	if not fitsl: ds1=sl0-1e-6; ds2=sl0+1e-6
	if not fitaf: df1=af0-1e-6; df2=af0+1e-6
	cf,vm = curve_fit(maotf, data[:,0], data[:,1], p0=[1.0,wd0,0.0,sl0,af0, 0.0,0.0,0.0], bounds=[[0.5,dw1,-5,ds1,df1, -0.1,-0.1,-0.5],[1.5,dw2,5,ds2,df2, 0.1,0.1,0.5]])
	model = maotf(data[:,0],*cf)
	print('%.1f %d %.2f %.2f %.2f' % (line,lorder,cf[1],cf[3],cf[4]))

	plt.figure(figsize=[11,6])
	plt.title('Line: %d cm-1 - Order %d' % (line,lorder))
	plt.plot(data[:,0],data[:,1],linewidth=0.7)
	plt.plot(data[:,0],model,linewidth=2)
	plt.plot(data[:,0],msinc,linewidth=2)
	plt.xlim([-150,150])
	plt.xlabel('Frequency [cm-1]')
	lbl  = 'Width: %.2f\n' % cf[1]
	lbl += 'Sidelobes: %.2f\n' % cf[3]
	lbl += 'Asymmetry: %.2f\n' % cf[4]
	plt.text(-140, 0.8, lbl, fontsize='12')
	for i in range(-8,8):
		xp = i*lblaze + cf[2]; ls=':'
		if i in cc: ls='-'
		plt.plot([xp,xp],[0,1],linestyle=ls,color='black')
	#Endfor
	plt.tight_layout()
	plt.savefig('aotf/aotf_%d_fit.png' % line)
#Endfor
