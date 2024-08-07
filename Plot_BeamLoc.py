#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 23:07:36 2024

@author: u1318104
"""

#%% load data
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os

#change to the dirtectory that data is saved 
dir_data='/uufs/chpc.utah.edu/common/home/flin-group4/qicheng/Meadow/SAC_v2/stack_all_Drift/'
dir_results=dir_data
Stas=np.loadtxt(dir_data+'station_line.lst',
                dtype=str,usecols=(0),unpack=True)
Lats,Lons=np.loadtxt(dir_data+'station_line.lst',
                dtype=float,usecols=(2,3),unpack=True)
#%% load data
os.chdir(dir_results)
fp_Vels=glob('Results_p25_vel*.txt')
fp_Vels.sort()
Vels=np.empty(len(fp_Vels))
Beam=np.empty(len(fp_Vels))
for i in range(len(fp_Vels)):
    Vels[i]=int(fp_Vels[i].split('.')[0][-3:])


for i in range(len(fp_Vels)):
    results=np.loadtxt(fp_Vels[i],unpack=True)
    Beam[i]=np.max(results)
#%% maximum Beam Power as function of search velocity
plt.figure()
plt.plot(Vels,Beam,'.--')
plt.xlabel('Speed (m/s)')
plt.ylabel('Beamm Power')
dir_fig=dir_data
# plt.savefig(dir_fig+'BeamVel.png',dpi=300)
#%%
LonsB=np.arange(-112.515,-112.47,0.001)
LatsB=np.arange(38.832,38.88,0.001)

Lons2d,Lats2d=np.meshgrid(LonsB,LatsB)
Lats2d,Lons2d=np.meshgrid(LatsB,LonsB)

vel=340
results=np.loadtxt('Results_p25_vel%d.txt'%vel,unpack=True)
results0=np.array(results.copy())
results0=results0.reshape(Lons2d.shape)
plt.figure()
plt.contourf(Lons2d,Lats2d,results0,np.linspace(0,np.max(results0),101),cmap='jet')
# plt.xlim([-112.506,-112.47])
# plt.ylim([38.840,38.875])
plt.colorbar()
plt.xticks(np.arange(-112.51,-112.47,0.01))
plt.yticks(np.arange(38.840,38.870+1e-5,0.01))

plt.plot(Lons,Lats,'k^')
# plt.title('%.3f,%.3f'%(Lons2d.reshape(-1)[np.argmax(results0)],Lats2d.reshape(-1)[np.argmax(results0)]))

dir_fig=dir_data
# plt.savefig(dir_fig+'BeamLoc%d_v0.png'%vel,dpi=300)