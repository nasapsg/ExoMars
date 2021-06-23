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
import os
import numpy as np
import matplotlib.pyplot as plt

files = os.listdir('aotf'); files.sort();
fs=[]; ws=[]; ss=[]; aa=[]; os=[]
for file in files:
    if file[0:4]!='fit_': continue
    fr = open('aotf/%s' % file,'r'); ln = fr.readline(); fr.close(); vs=ln.split()
    fs.append(float(vs[0])); ws.append(float(vs[1])); ss.append(float(vs[2]));
    aa.append(float(vs[3])); os.append(float(vs[4]))
#Endfor

# Perform the fits
aotfwc = np.polyfit(fs, ws, 2); print(aotfwc)
aotfsc = np.polyfit(fs, ss, 2); print(aotfsc)
aotfac = np.polyfit(fs, aa, 2); print(aotfac)
aotfoc = np.polyfit(fs, os, 2); print(aotfoc)

# Fit/plot the results
pl,ax = plt.subplots(1,4, figsize=(14, 4))
ax[0].plot(fs,ws,'o',label='Data')
ax[0].plot(fs,np.polyval(aotfwc,fs),label='Fit')
ax[0].set_xlabel('AOTF frequency [cm-1]')
ax[0].set_ylabel('AOTF width [cm-1]')
ax[0].legend()

ax[1].plot(fs,ss,'o',label='Data')
ax[1].set_xlabel('AOTF frequency [cm-1]')
ax[1].plot(fs,np.polyval(aotfsc,fs),label='Fit')
ax[1].set_ylabel('Sidelobes factor')
ax[1].legend()

ax[2].plot(fs,aa,'o',label='Data')
ax[2].set_xlabel('AOTF frequency [cm-1]')
ax[2].plot(fs,np.polyval(aotfac,fs),label='Fit')
ax[2].set_ylabel('Asymmetry')
ax[2].legend()

ax[3].plot(fs,os,'o',label='Data')
ax[3].set_xlabel('AOTF frequency [cm-1]')
ax[3].plot(fs,np.polyval(aotfoc,fs),label='Fit')
ax[3].set_ylabel('Offset')
ax[3].legend()

plt.tight_layout()
plt.savefig('aotf/fits.png')
plt.show()
