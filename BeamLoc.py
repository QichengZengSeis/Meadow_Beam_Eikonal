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

###all the stations
Stas=np.loadtxt(dir_data+'station_line.lst',
                dtype=str,usecols=(0),unpack=True)
Lats,Lons=np.loadtxt(dir_data+'station_line.lst',
                dtype=float,usecols=(2,3),unpack=True)

#%% get elevation? not needed if station distance is much smaller than distance to local source
'''
import urllib
import requests

LonsBE=Lons#np.arange(-112.50,-112.47+1e-5,0.001)
LatsBE=Lats#np.arange(38.83,38.88+1e-5,0.001)
ElvsBE=np.empty(LonsBE.shape)
url='https://api.open-elevation.com/api/v1/lookup?'

for i in tqdm(range(len(LonsBE))):
    lon=LonsBE[i]
    lat=LatsBE[i]
    
    params={'locations':f"{lat},{lon}"}
    result=requests.get((url+urllib.parse.urlencode(params)))
    ElvsBE[i]=result.json()['results'][0]['elevation']
'''
#%%
# plt.figure()
# plt.plot(Lons,Lats,'r*')
# for i in range(len(Stas)):
#     plt.text(Lons[i],Lats[i],Stas[i])
#%% Time Shift in time = phase shift in frequency
def TimeShiftFre(xt,tt,t0):
    #positive advance
    #negative delay
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

#%%
vmin=0.1 #km/s
vmax=2 #km/s
def func_BeamLoc(per,lonB,latB,slow_t,Stnm_s,iPara):
    # print(1,timeit.default_timer())
    eps=1e-5
    # tt=np.arange(-200,200+eps,0.1)
    tt=np.arange(-60,60+eps,0.01)
    
    DistA=[]
    XtA0=np.zeros(tt.shape)
    
    for iSta_s in tqdm(range(len(Stas))):

        sta_s=Stas[iSta_s]
        lat_s=Lats[iSta_s]
        lon_s=Lons[iSta_s]
        # if not sta_s == Stnm_s:
        #     continue
        # print(sta_s)        
    
        # for iSta_r in tqdm(range(len(Stas))):
        for iSta_r in range(len(Stas)):
        # for iSta_r in [1]:
            # if iSta_s==iSta_r:
            #     continue
            if iSta_s>iSta_r:
                continue
        # for iSta_r in range(10):
            sta_r=Stas[iSta_r]
            lat_r=Lats[iSta_r]
            lon_r=Lons[iSta_r]
            
            # distance between beam center and source station
            distBs_m, azBs, bazBs = gps2dist_azimuth(latB,lonB,lat_s,lon_s) #cross-correlation, source-receiver: dist, az, baz
            distBs_km=distBs_m/1000  
            
            # distance between beam center and receiver station
            distBr_m, azBr, bazBr = gps2dist_azimuth(latB,lonB,lat_r,lon_r) #cross-correlation, source-receiver: dist, az, baz
            distBr_km=distBr_m/1000  
            
            # distance between source station and receiver station
            distsr_m, azsr, bazsr = gps2dist_azimuth(lat_s,lon_s,lat_r,lon_r) #cross-correlation, source-receiver: dist, az, baz
            distsr_km=distsr_m/1000  
            
            # only uisng nearby stations, waveforms are distorted, especially at SE of gas line
            if distsr_km>0.25:
                continue
            
            try:    
                st=read(dir_data+'ZZ/COR_'+sta_s+'-'+sta_r+'_ZZ_stacked.SAC_2-8Hz')
            except:
                continue
            DistA.append(distsr_km)
            
            xt=st[0].data/np.max(np.abs(st[0].data)) #Normalize by max abs
            
            # for ibaz in np.arange(len(baz_arr)):
                # baz_t=baz_arr[ibaz]
                # time_delay=dist_km*cos((baz-baz_t)/180*np.pi)*slow_t
            time_delay=(distBr_km-distBs_km)*slow_t

            xtNew=TimeShiftFre(xt,tt,time_delay)
            XtA0=XtA0+xtNew
            # print(6,timeit.default_timer())
                # XtA1[ibaz,:]=XtA1[ibaz,:]+xtNew
    # XtA0=np.array(XtA0)
    # XtA1=np.array(XtA1)
    # StackA1=np.sum(XtA1,axis=0)
    if len(DistA)>0:
        StackA0=XtA0/len(DistA)
    else:
        StackA0=XtA0
    
    env=hilbert(StackA0)
    env_max=np.max(np.abs(env))
    return env_max

#%%
eps=1e-5
Stnm_s='001'
per=0.2 #not used in this version

#vel in meters per second
for vel in np.array([340]):#np.array([335,336,337,338,339,340,341,342,343,345,300,305,310,315,320,325,330,350]):
    Para_Pool=[]
    slow_t=1/(vel/1000) #sound speed 340 m/s
    
    #Beam Center search Range
    LonsB=np.arange(-112.515,-112.47,0.001)
    LatsB=np.arange(38.832,38.88,0.001)
    
    iPara=0
    for lonB in LonsB: #tmp/search s/km
        for latB in LatsB:
        # for baz_t in baz_arr: #tmp/search deg
            iPara+=1
            Para_Pool.append((per,lonB,latB,slow_t,Stnm_s,iPara))
    
    pool=Pool(120)
    results=pool.starmap(func_BeamLoc,Para_Pool)
    pool.close()
    pool.join()
    
    #modify save directory if needed
    dir_save=dir_data
    np.savetxt(dir_save+'Results_p25_vel%d.txt'%(1/slow_t*1000),results,fmt='%.8f')


