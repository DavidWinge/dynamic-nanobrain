#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 09:29:28 2022

@author: dwinge
"""


import numpy as np
from matplotlib import pyplot as plt

# Define parameters
my_dpi = 300
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

# Figure sizes
inchmm = 25.4
nature_single = 89.0 / 25.4
nature_double = 183.0 / 25.4
nature_full = 247.0 / 25.4

# Plot options
font = {'family' : 'sans',
        'weight' : 'normal',
        'size'   : 14}
plt.rc('font', **font)


#%% Plot the Vds sweep


# Source-drain sweep plot
# Finger contacts
I_VDS_VG = np.loadtxt('../parameters/FET_Vsd_Vg_Lg200nm.txt',comments='%')
    
VG = np.arange(0.,0.55,step=0.1)
print(VG)

figname = 'sweep_Vds'

fig, ax = plt.subplots()#figsize=(nature_single,nature_single))

for k in range(0,len(I_VDS_VG[0,:])-1) :  
    ax.plot(I_VDS_VG[:,0],I_VDS_VG[:,k+1],label=f'{VG[k]:.1f} V', color=colors[k])

ax.grid(True)

#ax.set_yscale('log')
ax.set_xlabel('Vds (V)')
ax.set_ylabel('I (uA)')
ax.set_ylim(0,18)
ax.set_xlim(0,1.0)
#ax.set_title('Source-drain sweep with mobility model on')

plt.tight_layout()
plt.legend(loc='upper left')
#plt.savefig(figname+'.eps',bbox_inhces='tight')
plt.savefig(figname+'.png',dpi=my_dpi)
#plt.savefig(figname+'.svg')
plt.show()


#%% Now we create the interpolation map
from scipy.interpolate import RegularGridInterpolator

VDS = I_VDS_VG[:,0]
X, Y = np.meshgrid(VG,VDS)
IDS = I_VDS_VG[:,1:]

inter_2D = RegularGridInterpolator((VDS,VG),IDS)


grid_VDS, grid_VG = np.mgrid[0:0.8:50j,0.0:0.5:11j]

inter_vals = inter_2D((grid_VDS.ravel(),grid_VG.ravel()))

inter_vals = inter_vals.reshape((grid_VDS.shape[0],grid_VG.shape[1]))

fig, ax = plt.subplots(figsize=(nature_single,nature_single))

for k in range(0,len(I_VDS_VG[0,:])-1) :  
    line, = ax.plot(I_VDS_VG[:,0],I_VDS_VG[:,k+1],label=VG[k])
    #ax.plot(I_VDS_pad_MMp5[:,0],I_VDS_pad_MMp5[:,k],'--',color=colors[1])
    #if k==1 : line.set_label(r'$V_G$=-0.5,-0.25,0.0,0.25 V')
    #Ids = myFET.Ids(VG[k-1],Vds)
    #ax.plot(Vds,Ids,'k--')
    
for k in range(0,grid_VDS.shape[1]) :
    ax.plot(grid_VDS[:,0],inter_vals[:,k],'--',label=grid_VG[0,k])


ax.legend()
plt.show()

#%% Now we try to get the diff resistance as well

I_LED = np.loadtxt('../parameters/LED_iv_s1e3-s1e6.txt',comments='%')

I_S1E4 = I_LED[56:111,1]
VLED = I_LED[56:111,0]

# Calculate vector holding the diff resistance
delta_I = I_S1E4[1:]-I_S1E4[:-1]

# Interpolate for I(VLED)
from scipy.interpolate import interp1d
intp_iv_LED = interp1d(VLED,I_S1E4,kind='cubic',fill_value='extrapolate')

vspace = np.linspace(0.6,1.83,301)
dv = vspace[1]-vspace[0]
delta_I = intp_iv_LED(vspace[1:])-intp_iv_LED(vspace[:-1])
dI_dV = delta_I/dv



#%% Construct the differential resistance
intp_dR_LED = interp1d(vspace[1:],dI_dV**-1)


#%%

fig, ax = plt.subplots()

ax.plot(VLED,I_S1E4)
ax.plot(vspace,intp_iv_LED(vspace),'--')

ax.set_yscale('linear')

plt.show()

#%%

fig, ax = plt.subplots()

ax.plot(vspace[1:],intp_dR_LED(vspace[1:]))

plt.show()