# -------------------------------------------------------------
# Calibration procedure to determine the number of orders needed
# Developed for the ExoMars/TGO/NOMAD instrument
# Villanueva, Liuzzi - NASA/GSFC - April 2021
# -------------------------------------------------------------
# This script takes the recipe of the AOTF and grating and determines
# the numbers of order needed when modeling/analyzing data for a specific
# order and altitude. The script synthesizes spectra using PSG, and assumes
# a +/- 10 to be the reference spectrum.
# To run type 'python norder.py'
# -------------------------------------------------------------
import os, sys
import numpy as np
import matplotlib.pyplot as plt

# https://nomad.aeronomie.be/index.php/cop-tables/commonly-measured-diffraction-orders
orders = [121,134,148,168,169,191]
alts = [5,25,50,75,100]
maxSNR = 500.0

# AOTF shape parameters and grating
aotfwc  = [1.11085173e-06, -8.88538288e-03,  3.83437870e+01] # Sinc width [cm-1 from AOTF frequency cm-1]
aotfsc  = [2.87490586e-06, -1.65141511e-02,  2.49266314e+01] # sidelobes factor [scaler from AOTF frequency cm-1]
aotfaf  = [-5.47912085e-07, 3.60576934e-03, -4.99837334e+00] # Asymmetry factor [scaler from AOTF frequency cm-1]
cfpixel = np.array([1.75128E-08, 5.55953E-04, 2.24734E+01])  # Blaze free-spectral-range (FSR) [cm-1 from pixel]

# PSG parameters
server = 'https://psg.gsfc.nasa.gov'
fr = open("spectra/norder_psg.txt", "r"); cfg = fr.readlines(); fr.close()

# Define AOTF sinc
def sinc(dx,amp,width,lobe,asym):
	sinc = amp*(width*np.sin(np.pi*dx/width)/(np.pi*dx))**2
	ind = (abs(dx)>width).nonzero()[0]
	if len(ind)>0: sinc[ind] = sinc[ind]*lobe
	ind = (dx<=-width).nonzero()[0]
	if len(ind)>0: sinc[ind] = sinc[ind]*asym
	return sinc
#Endif sinc

#Iterate across orders
for order in orders:
	# Define spectral range for +/- 10 orders
	blaze = np.polyval(cfpixel,160.0) # Free-spectral-range (FSR) of the grating [cm-1] at pixel=160
	f0 = blaze*order
	fmin = f0 - blaze*11
	fmax = f0 + blaze*11

	# Parameters needed for performing the order addition (these are fed to PSG)
	blaze0 = f0                       # Center of the blaze [cm-1]
	blazeFSR = blaze                  # Free-spectral-range (FSR) of the grating [cm-1]
	AOTF0 = f0                        # Center of the AOTF [cm-1], for this excercise blaze=AOTF
	AOTFw = np.polyval(aotfwc,AOTF0)  # Width of the AOTF [cm-1], distance to first zero-crossing
	AOTFs = np.polyval(aotfsc,AOTF0)  # Sidelobe ratio of the AOTF (1.0 is classical sinc)
	AOTFa = np.polyval(aotfaf,AOTF0)  # Asymmetry in the sidelobe ratio intensity (lower/upper, 1.0 is classical sinc)

	# Iterate across altitudes
	for alt in alts:
		fspec = 'spectra/psg_%d_%d.txt' % (order,alt)
		if not os.path.isdir('spectra'): os.system('mkdir spectra')
		if not os.path.exists(fspec):
			print('Generating %s' % fspec)
			cfg[0] = '<GENERATOR-RANGE1>%.2f\n' % fmin
			cfg[1] = '<GENERATOR-RANGE2>%.2f\n' % fmax
			cfg[2] = '<GEOMETRY-USER-PARAM>%.2f\n' % alt
			fr = open("spectra/norder_cfg.txt", "w")
			for line in cfg: fr.write(line)
			fr.close()
			os.system('curl -s --data-urlencode file@spectra/norder_cfg.txt %s/api.php > %s' % (server,fspec))
		#End generating radiances
		spec = np.genfromtxt(fspec)

		# Perform the order addition
		aotf = sinc(spec[:,0]-AOTF0, 1.0, AOTFw, AOTFs, AOTFa)
		f1 = np.polyval(cfpixel,  0)*order; i2 = np.searchsorted(-spec[:,0],-f1)
		f2 = np.polyval(cfpixel,320)*order; i1 = np.searchsorted(-spec[:,0],-f2)
		freq = spec[i1:i2+1,0]
		orders = np.zeros([2,21,i2-i1+1])
		fblaze = (blazeFSR*np.sin(np.pi*(freq-blaze0)/blazeFSR)/(np.pi*(freq-blaze0)))**2
		for io in range(-10,10+1):
			# In PSG, if the selected spectral resolution is RP (resolving power)
			# the spectral sampling is proportional to the frequency/wavelength.
			# This is the natural sampling of the grating system. If one chooses
			# a convolution at 20,000, then PSG samples at 10x that (200,000)
			# A specific pixel shift defines the offset between frequency from orders
			nsh=round(np.log((blaze0+io*blazeFSR)/blaze0)/np.log((1.0+(1.0/(10.0*20000.0)))))
			for i in range(i1,i2+1):
				orders[0,io+10,i-i1] = fblaze[i-i1]*aotf[i-nsh]
				orders[1,io+10,i-i1] = fblaze[i-i1]*aotf[i-nsh]*spec[i-nsh,1]
			#Endfor
		#Endfor

		# Analyze the order addition
		pl,ax = plt.subplots(2,1, sharex=True, figsize=(9, 7))
		st = np.sum(orders,axis=1); tflux = np.sum(st[1,:]); tspec = st[1,:]/st[0,:]
		spec = orders[:,10,:]; nord=1
		ax[0].plot(freq,spec[1,:]/spec[0,:],linestyle=':',color='black',linewidth=0.5,label='Non-AOTF')
		for io in range(1,10+1):
			spec += orders[:,10-io,:]
			spec += orders[:,10+io,:]
			ts = spec[1,:]/spec[0,:]
			tf = np.sum(spec[1,:])
			me = np.max(abs(ts-tspec))
			if me>(1.0/maxSNR): nord=io
			ax[0].plot(freq,ts,label='+/- %d' % io)
			ax[1].plot(freq,ts-tspec,label='+/- %d' % io)
		#Endfor
		ax[0].set_xlim([f1,f2])
		ax[0].set_title('Order %d - Altitude %d km - Need +/- %d for SNR=%d' % (order,alt,nord,maxSNR))
		ax[0].set_ylabel('Normalized intensity')
		ax[1].set_ylabel('Residuals')
		ax[1].set_xlabel('Frequency [cm-1]')
		ax[0].legend()
		plt.tight_layout()
		plt.savefig('spectra/psg_%d_%d.png' % (order,alt))
		plt.close()
		print('%3d %3d %2d' % (order,alt,nord))
	#Endfor
#Endfor
