#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 11:59:12 2022

@author: u1318104
"""

import numpy as np
from numpy import cos, pi, ceil
import matplotlib.pyplot as plt
from obspy.geodetics.base import gps2dist_azimuth
from obspy import read, Trace 
from math import log

from scipy.fft import fft, ifft
from scipy.signal import hilbert
from multiprocessing import Pool

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from tqdm import tqdm
import timeit

############ modify as needed
dir_data='/uufs/chpc.utah.edu/common/home/flin-group4/qicheng/Meadow/SAC_v2/stack_all_Drift/'

#NW subset of the array, avoid waveform distortion by topography
Stas=np.loadtxt(dir_data+'station_line_Beam2.lst',
                dtype=str,usecols=(0),unpack=True)
Lats,Lons=np.loadtxt(dir_data+'station_line_Beam2.lst',
                dtype=float,usecols=(2,3),unpack=True)
#%
# plt.figure()
# plt.plot(Lons,Lats,'r*')
# for i in range(len(Stas)):
#     plt.text(Lons[i],Lats[i],Stas[i])
#%% Time Shift in time = phase shift in frequency
def TimeShiftFre(xt,tt,t0):
    # t0 postive advance
    # t0 negative delay
    N=int(2**ceil(log(len(xt)*2,2)))
    dt=tt[1]-tt[0]
    df=1/(N*dt)
    ff=np.arange(0,N*df,df)
    
    xtpad=np.zeros(N)
    xtpad[0:len(xt)]=xt
    Xfpad=fft(xtpad)
    Coe=np.exp(2j*pi*ff*t0)
    XfNewpad=Xfpad*Coe
    xtNewpad=np.real(ifft(XfNewpad))
    xtNew=xtNewpad[0:len(xt)]
    
    return xtNew

# tt=np.arange(0,10,0.1)
tt=np.arange(-60,60+1e-5,0.01)
xt=np.zeros(tt.shape)
xt[6000]=1
xtNew=TimeShiftFre(xt,tt,20)
# plt.figure()
# plt.plot(tt,xt,'b.-')    
# plt.plot(tt,xtNew,'r.--')

#%% Beamforming function for time delay shift depending on local slowness and source back azimuth
# vmin=0.1 #km/s
# vmax=2 #km/s
def func_Beam(per,slow_t,baz_arr,Stnm_s):
    eps=1e-5
    # tt=np.arange(-200,200+eps,0.1)
    tt=np.arange(-60,60+eps,0.01)

    DistA=[]
    XtA0=np.zeros(tt.shape) #store stacked waveform
    XtA1=np.zeros((len(baz_arr),len(tt))) #stacked waveforms, array of back azimuth 
    
    for iSta_s in range(len(Stas)):
    # for iSta_s in tqdm(range(2)):

        sta_s=Stas[iSta_s]
        lat_s=Lats[iSta_s]
        lon_s=Lons[iSta_s]
        # if not sta_s == Stnm_s:
        #     continue
        print(sta_s)        
    
        for iSta_r in tqdm(range(len(Stas))):
        # for iSta_r in range(1):
            if iSta_s>iSta_r:
                continue
        # for iSta_r in range(10):
            sta_r=Stas[iSta_r]
            lat_r=Lats[iSta_r]
            lon_r=Lons[iSta_r]

            cri_dis=per/slow_t #farfield 1 wavelength? #/2 #farfield criteria - half wavelength
            dist_m, az, baz = gps2dist_azimuth(lat_s,lon_s,lat_r,lon_r) #cross-correlation, source-receiver: dist, az, baz
            dist_km=dist_m/1000         
            
            ########### Distance Criteria
            # if dist_km<cri_dis: # farfield criteria - half wavelength
            #     continue
            # cri_dis1=1.9
            # if dist_km>cri_dis1: # farfield criteria - half wavelength
            #     continue
            
            # cri_dis0=0.5
            # if dist_km<cri_dis0: # farfield criteria - half wavelength
            #     continue
        
            # if dist_km>0.25:
            #     continue
            
            try:    
                st=read(dir_data+'ZZ/COR_'+sta_s+'-'+sta_r+'_ZZ_stacked.SAC_2-8Hz') #filtered for 
            except:
                continue
            DistA.append(dist_km)
            
            ######################################################################
            # t_taper0=dist_km/vmax
            # # t_taper1=dist_km/vmin
            # dt=0.01

            # TW=np.ones(tt.shape)
            # tw_min=1 #tapering width on one side, in secs
            # if t_taper0/dt>=tw_min:
            #     for it in range(len(tt)):
            #         if np.abs(tt[it])<=t_taper0 and np.abs(tt[it])>=t_taper0-tw_min:
            #             TW[it]=(1-np.cos((np.abs(tt[it])-(t_taper0-tw_min))/tw_min*np.pi))/2
            #         elif np.abs(tt[it])<t_taper0-tw_min:
            #             TW[it]=0
            #         else:
            #             continue
            # else:
            #     for it in range(len(tt)):
            #         if np.abs(tt[it])<=t_taper0:
            #             TW[it]=(1-np.cos(np.abs(tt[it])/t_taper0*np.pi))/2
            # st[0].data=st[0].data*TW
            ######################################################################
            
            # st.filter('bandpass',freqmin=1./per/1.2,freqmax=1./per/0.8,corners=4,zerophase=True)
            # st.filter('bandpass',freqmin=2,freqmax=8,corners=4,zerophase=True)

            xt=st[0].data/np.max(np.abs(st[0].data)) #Normalize by max abs
            XtA0=XtA0+xt
            
            for ibaz in np.arange(len(baz_arr)):
                baz_t=baz_arr[ibaz]
                time_delay=dist_km*cos((baz-baz_t)/180*np.pi)*slow_t
                xtNew=TimeShiftFre(xt,tt,time_delay)
            
                XtA1[ibaz,:]=XtA1[ibaz,:]+xtNew
    
    if len(DistA)>0:
        StackA1=XtA1/len(DistA)
    else:
        StackA1=XtA1
    
    env_max=np.zeros(len(baz_arr))
    for i in range(len(baz_arr)):
        env=hilbert(StackA1[i,:])
        env_max[i]=np.max(np.abs(env))
        
    # imax=np.argmax(np.abs(env))
    # print()
    return env_max


#%% Multiprocessing parallel due to python GIL
eps=1e-5
Stnm_s='003'

per=0.2 #not doing anything in this version

Para_Pool=[]
  
slow_arr=np.arange(0.5,5.0,0.1) #slowness search in s/km
baz_arr=np.arange(0,360,2) #source backazimuth in degrees

for slow_t in slow_arr: #tmp/search s/km
    # for baz_t in baz_arr: #tmp/search deg
    Para_Pool.append((per,slow_t,baz_arr,Stnm_s))

pool=Pool(50)
results=pool.starmap(func_Beam,Para_Pool) #results stored sequentially consistent with Para_Pool input
pool.close()
pool.join()


baz2d,slow2d=np.meshgrid(baz_arr,slow_arr)
results0=np.array(results.copy())
results0=results0.reshape((len(slow_arr),len(baz_arr)))

#%% Plotting
fig = plt.figure()
ax = Axes3D(fig)

# rad = np.linspace(0, 5, 100)
# azm = np.linspace(0, 2 * np.pi, 100)
# r, th = np.meshgrid(rad, azm)
# z = (r ** 2.0) / 4.0

plt.subplot(projection="polar")

plt.pcolormesh(baz2d.T/180*pi, slow2d.T, results0.T,cmap='jet')
#plt.pcolormesh(th, z, r)

# plt.plot(azm, r, color='k', ls='none') 
plt.grid()
plt.colorbar()
# plt.show()
plt.savefig('/uufs/chpc.utah.edu/common/home/u1318104/Figures/Zanskar_Meadow/08062024_RD/Beam2_Freq2-8Hz_'+str(per)+'s_daz2_dslowp1_sta%sAll.png'%Stnm_s,dpi=300)

    
    # plt.savefig('/uufs/chpc.utah.edu/common/home/u1318104/Figures/Zanskar_Meadow/06212024_Beam/'+str(per)+'s_daz10_dslowp1.eps')
    # np.savetxt(dir_data+'../'+str(per)+'daz1_sta%s.dat'%Stnm_s,results0,fmt='%.4f')
#%%


        
        


