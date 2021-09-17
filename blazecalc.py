# -------------------------------------------------------------
# Developed for the ExoMars/TGO/NOMAD SO instrument
# September 2021
# -------------------------------------------------------------
import os
import numpy as np
import matplotlib.pyplot as plt

# AOTF shape parameters
aotfwc  = [-1.78088527e-07,  9.44266907e-04,  1.95991162e+01] # Sinc width [cm-1 from AOTF frequency cm-1]
aotfsc  = [ 1.29304371e-06, -6.77032965e-03,  1.03141366e+01] # sidelobes factor [scaler from AOTF frequency cm-1]
aotfac  = [-1.96949242e-07,  1.48847262e-03, -1.40522510e+00] # Asymmetry factor [scaler from AOTF frequency cm-1]
aotfoc  = [            0.0,             0.0,             0.0] # Offset [coefficients for AOTF frequency cm-1]
aotfgc  = [ 1.07865793e-07, -7.20862528e-04,  1.24871556e+00] # Gaussian peak intensity [coefficients for AOTF frequency cm-1]

# Calibration coefficients
cfaotf  = np.array([1.34082e-7, 0.1497089, 305.0604])         # Frequency of AOTF [cm-1 from kHz]
cfpixel = np.array([1.75128E-08, 5.55953E-04, 2.24734E+01])   # Blaze free-spectral-range (FSR) [cm-1 from pixel]
ncoeff  = [-1.76520810e-07, -2.26677449e-05, -1.93885521e-04] # Relative frequency shift coefficients [shift/frequency from Celsius]
blazep  = [-5.76161e-14,-2.01122e-10,2.02312e-06,2.25875e+01] # Dependence of blazew from AOTF frequency
aotfts  = -6.5278e-5                                          # AOTF frequency shift due to temperature [relative cm-1 from Celsius]
norder  = 6                                                   # Number of +/- orders to be considered in order-addition

# Input parameters
aotf  = 19907   # AOTF Frequency [kHz]
tempg = 0.0     # Temperature for the grating [C]
tempa = -3.0    # Temperature for the AOTF [C]

# Calculate blaze parameters
aotff = np.polyval(cfaotf, aotf) + tempa*aotfts  # AOTF frequency [cm-1], temperature corrected
blazew =  np.polyval(blazep,aotf-22000.0)        # FSR (Free Spectral Range), blaze width [cm-1]
blazew += blazew*np.polyval(ncoeff,tempg)        # FSR, corrected for temperature
order = round(aotff/blazew)                      # Grating order
blazef = order*blazew                            # Center of the blaze

# Compute AOTF parameters
aotfw = np.polyval(aotfwc,aotff)
aotfs = np.polyval(aotfsc,aotff)
aotfa = np.polyval(aotfac,aotff)
aotfo = np.polyval(aotfoc,aotff)
aotfg = np.polyval(aotfgc,aotff)

# Frequency of the pixels
pixf  = np.polyval(cfpixel,range(320))*order
pixf += pixf*np.polyval(ncoeff, tempg)

print("Order: %d" %order)
print("AOTF parameters [center, width, sidelobes, asymmetry, gaussian]: %.4f %.4f %.4f %.4f %.4f" %(aotff,aotfw,aotfs,aotfa,aotfg))
print("Blaze parameters [center, width]: %.4f %.4f" %(blazef,blazew))
print("Frequency range of order: %.4f %.4f" %(pixf[0],pixf[-1]))
