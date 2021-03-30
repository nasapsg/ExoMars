# -------------------------------------------------------------
# Calibration procedure to derive the "recipe" for the AOTF filter
# Developed for the ExoMars/TGO/NOMAD instrument
# Villanueva, Liuzzi - NASA/GSFC - April 2021
# -------------------------------------------------------------
# Before running this code ('python fits.py'), run the aotf.py
# script and save the outputs of that code into a file (e.g., 'fits_a.txt')
# This code takes the different shape fits across orders, and
# performs a linear (2nd order) regression fit to the parameters
# deriving a "recipe" applicable to other orders
# -------------------------------------------------------------
# fits_a.txt: parameters with all parameters (width, sidelobe, asymmetry) free
# fits_b.txt: parameters with locking width and sidelobe/asymmetry free
# fits_c.txt: parameters with locking width/sidelobe and asymmetry free
# fits_d.txt: parameters with locking sidelobe/asymmetry and width free
# -------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

# Old recipe (Aug/2019)
aotfwc  = [-2.18387e-7,5.82007e-4,21.8543]  # Sinc width
aotfoc  = [-3.51707e-8,2.80952e-4,-0.4997]  # Baseline (offset)
aotfsc  = [4.25071e-7,-2.24849e-3,4.24031]  # sidelobes factor

# Fit/plot the results
pl,ax = plt.subplots(1,3, figsize=(13, 4))
data  = np.genfromtxt('aotf/fits_d.txt')
aotfw = np.polyval(aotfwc,data[:,0])
cfitw = np.polyfit(data[:,0],data[:,2], 2)
fitw  = np.polyval(cfitw,data[:,0])
ax[0].plot(data[:,0],data[:,2],'o',label='Data')
ax[0].plot(data[:,0],aotfw,label='Old')
ax[0].plot(data[:,0],fitw,label='New')
ax[0].set_xlabel('AOTF frequency [cm-1]')
ax[0].set_ylabel('AOTF width [cm-1]')
ax[0].legend()
print(cfitw)

data  = np.genfromtxt('aotf/fits_b.txt')
aotfs = np.polyval(aotfsc,data[:,0])
cfits = np.polyfit(data[:,0],data[:,3], 2)
fits  = np.polyval(cfits,data[:,0])
ax[1].plot(data[:,0],data[:,3],'o',label='Data')
ax[1].plot(data[:,0],fits,label='New')
ax[1].plot(data[:,0],data[:,3]*(1.0+data[:,4])/2.0,'o',label='Data Left+Right')
ax[1].plot(data[:,0],aotfs,label='Old')
ax[1].set_xlabel('AOTF frequency [cm-1]')
ax[1].set_ylabel('Sidelobes factor')
ax[1].legend()
print(cfits)

data  = np.genfromtxt('aotf/fits_c.txt')
cfita = np.polyfit(data[:,0],data[:,4], 2)
fita  = np.polyval(cfita,data[:,0])
ax[2].plot(data[:,0],data[:,4],'o',label='Data')
ax[2].plot(data[:,0],fita,label='New')
ax[2].set_xlabel('AOTF frequency [cm-1]')
ax[2].set_ylabel('Asymmetry')
ax[2].legend()
print(cfita)

plt.tight_layout()
plt.savefig('aotf/fits.png')
plt.show()
