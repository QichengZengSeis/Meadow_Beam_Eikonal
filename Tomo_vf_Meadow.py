#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 10:12:28 2024

@author: u1318104
"""


#%%
import numpy as np
import os
from glob import glob
import matplotlib 
import matplotlib.pyplot as plt
from obspy.geodetics.base import gps2dist_azimuth
from tqdm import tqdm
from multiprocessing import Pool
from numba import jit


from cartopy.io.img_tiles import GoogleTiles, Stamen#, StamenTerrain
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib
import cartopy.feature as cfeature


#%%


# if True:  
def MapWUS(Lon1,Lat1,DATA,VMIN,VMAX,CMNEW):
    xmax=-112.47+360 #-100+360
    ymax=38.875 #52
    xmin=-112.51+360 #-130
    ymin=38.84 #28

    fdir='/uufs/chpc.utah.edu/common/home/u1318104/Research/1Psi/'
    # fp=open(fdir+'wus_province_II.dat','r')
    # Physio=fp.read().split('99999 99999')
    
    # flag=1 #flag for lon
    # faultslon=list()
    # faultslat=list()
    # for fault in Physio:
    #     if fault=='':
    #         continue
    #     faultlonlat=fault.split()
    #     tmplon=list()
    #     tmplat=list()
    #     for tmp in faultlonlat:
    #         if flag:
    #             tmplon.append(float(tmp))
    #         else:
    #             tmplat.append(float(tmp))
    #         flag=1-flag
    #     faultslon.append(tmplon)
    #     faultslat.append(tmplat)
    
    #WUS
    lonticks=np.arange(xmin,xmax+0.001,0.01)
    latticks=np.arange(ymin,ymax+0.001,0.01)
    #AK
    #US
    # lonticks=np.arange(xmin,xmax,10)
    # latticks=np.arange(ymin,ymax,5)
    # lonticks=np.arange(-120,-70+1,10)
    # latticks=np.arange(25,50+1,5)
    
    # request=StamenTerrain()
    request=GoogleTiles(style='satellite')
    request.desired_tile_form='L'
    
    MarkerSize=12.5 #50
    # fig=plt.figure(figsize=(12, 8)) #, facecolor="none"
    
    # # create mercator projection
    # ax0 = plt.axes(projection=ccrs.Mercator())
    
    fig, ax0=plt.subplots(figsize=(8, 8),subplot_kw=dict(projection=request.crs)) #12 8
    
    ## Reset Zoom level ###
    # xmine=-112.5; xmaxe=-111.6; ymine=40.4; ymaxe=40.9
    xmine=xmin; xmaxe=xmax; ymine=ymin; ymaxe=ymax
    
    # xmine=-130; xmaxe=-100; ymine=20; ymaxe=455
                                
    
    ax0.cla()
    ax0.set_extent([xmine,xmaxe,ymine,ymaxe], crs=ccrs.PlateCarree())
    # ax0.add_image(request,4,cmap='terrain') #,cmap='gray'
    # ax0.add_image(request,4,cmap='gray') #,cmap='gray'
    
    # Tick labels etc
    ax0.set_xticks(lonticks, crs=ccrs.PlateCarree())
    ax0.set_yticks(latticks, crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(number_format='.3f', degree_symbol='')  # ,zero_direction_label=True)
    lat_formatter = LatitudeFormatter(number_format='.3f', degree_symbol='')
    ax0.xaxis.set_major_formatter(lon_formatter)
    ax0.yaxis.set_major_formatter(lat_formatter)
    ax0.xlabel_style = {'size': 10, 'color': 'gray'}
    ax0.ylabel_style = {'size': 10, 'color': 'gray'}  # 'weight':'bold'
           
     
    # ax0.set_xticks([-120,-118,-116],minor=True,crs=ccrs.PlateCarree())       
    # ax0.set_xticks([-112.4,-112.2,-112.0,-111.8],minor=True,crs=ccrs.PlateCarree())       
    # ax0.scatter(Stlos,Stlas,c='k',marker='^',transform=ccrs.PlateCarree())
    
    # ax0.plot(Stlos,Stlas,color='k',marker='^',markersize=5,linestyle='None',transform=ccrs.Geodetic())

    
    #plot faults...
    # for ii in np.arange(len(faultslon)):
    #     ax0.plot(faultslon[ii],faultslat[ii],linewidth=2,color='r',zorder=2,transform=ccrs.Geodetic()) #transform=ccrs.Geodetic() #linewidth=2.
    # for ii in np.arange(len(rift_lon)):
    #     if ii !=0:
    #         continue
    #     ax0.plot(rift_lon[ii],rift_lat[ii],linewidth=2,color='r',zorder=2,transform=ccrs.Geodetic()) #transform=ccrs.Geodetic() #linewidth=2.
    # for ii in np.arange(len(BD_lon)):
    #     ax0.plot(BD_lon[ii],BD_lat[ii],linewidth=2,color='r',zorder=2,transform=ccrs.Geodetic()) #transform=ccrs.Geodetic() #linewidth=2.
    

    #60s US
    im=ax0.contourf(Lon1,Lat1,DATA,np.linspace(VMIN,VMAX,101),cmap=CMNEW,zorder=1,extend='both',transform=ccrs.PlateCarree())  #
    cbar=fig.colorbar(im,orientation='horizontal',ticks=np.linspace(VMIN,VMAX,7),pad=0.08,fraction=0.046,shrink=0.5,format='%.2f')
    cbar.ax.set_xlabel('Phase Velocity (km/s)')
    # cbar.ax.set_xlabel('Uncertainty (km/s)')


    ax0.add_feature(cfeature.LAKES)
    # ax0.add_feature(cfeature.BORDERS)
    # ax0.add_feature(cfeature.STATES.with_scale('50m'),linewidth=0.1)



def get_valT(per,perA,valA,fill_value='end'):
    '''

    Parameters
    ----------
    per : int or float
        period to be interpolated.
    perA : list or array
        a list or array of periods.
    valA : TYPE
        DESCRIPTION.
    fill_value : TYPE, optional
        fill value outside of perA range, np.nan or 0 or default end. The default is 'end'.

    Returns
    -------
    val : TYPE
        DESCRIPTION.

    '''
    id1=np.nanargmin(perA)
    id2=np.nanargmax(perA)
    for idx in range(len(perA)):
        if perA[idx]<per and perA[idx]>perA[id1]:
            id1=idx
        if perA[idx]>per and perA[idx]<perA[id2]:
            id2=idx
    per1=perA[id1]
    val1=valA[id1]
    per2=perA[id2]
    val2=valA[id2]
    if per<per1:
        if fill_value=='end':
            val=val1
        else:
            val=fill_value
    elif per>per2:
        if fill_value=='end':
            val=val2
        else:
            val=fill_value
    else:
        val=val1+(val2-val1)/(per2-per1)*(per-per1)
    return val

def colormap():
    #load colormap
    c1=((125./255.),0.,0.)
    c2=((255./255.),0.,0.) #red
    c3=((255./255.),(255./255.),0.) #yellow
    c4=((200./255.),(200./255.),(200./255.)) #gray

    # c4=((230./255.),(230./255.),(230./255.)) #gray
    # c4=((255./255.),(255./255.),(255./255.)) #white'
    c5=((150./255.),(255./255.),(150./255.)) #lightgreen
    c6=((100./255.),(185./255.),(255./255.)) #cyan
    c7=((100./255.),0.,(170./255.)) #purple
    
    # c1=((125./255.),0.,0.)
    # c2=((255./255.),0.,0.) #red

    # c3=((255./255.),(255./255.),0.) #yellow
    # c4=((230./255.),(230./255.),(230./255.)) #white'
    # c5=((150./255.),(255./255.),(150./255.)) #lightgreen
    # c6=((100./255.),(185./255.),(255./255.)) #cyan
    # c7=((100./255.),0.,(170./255.)) #purple
    
    
    colors = [c1,c2,c3,c4,c5,c6,c7]  # black, red, yellow, white, lightgreen, cyan, purple

    cmap_name = 'my_list'
    cmnew = matplotlib.colors.LinearSegmentedColormap.from_list(cmap_name, colors, N=500)
    return cmnew
cmnew=colormap()

@jit(nopython=True)
def get_dist(lat1,lon1,lat2,lon2):

    #double theta,pi,temp;
    #double radius=6371;
    #pi=4.0*atan(1.0);
    radius=6371; pi=4.0*np.arctan(1.0)
      
    lat1=np.arctan(0.993277*np.tan(lat1/180.0*pi))*180.0/pi
    lat2=np.arctan(0.993277*np.tan(lat2/180.0*pi))*180.0/pi
      
    #lat1=atan(0.993277*tan(lat1/180*pi))*180/pi;
    #lat2=atan(0.993277*tan(lat2/180*pi))*180/pi;
      
    temp=np.sin((90.0-lat1)/180.0*pi)*np.cos(lon1/180.0*pi)*np.sin((90.0-lat2)/180.0*pi)*np.cos(lon2/180.0*pi)+\
    np.sin((90.0-lat1)/180.0*pi)*np.sin(lon1/180.0*pi)*np.sin((90.0-lat2)/180.0*pi)*np.sin(lon2/180.0*pi)+\
    np.cos((90.0-lat1)/180.0*pi)*np.cos((90-lat2)/180.0*pi)
    if lat1==lat2 and lon1==lon2:
        # print('same location', lat1, lon1, lat2, lon2)
        return 0
    if temp>1:
        print('setting temp to 1')
        # print(lat1,lon1,lat2,lon2)
        temp=1
    if temp<-1:
        print('setting temp to -1')
        temp=-1
    
    theta=np.abs(np.arccos(temp));
    return theta*radius


def get_good_measurements(Pers,event,Fps,Stlo,Stla,Dists,dir_data,dir_tomo):
    '''
    

    Parameters
    ----------
    per : TYPE
        DESCRIPTION.
    event : TYPE
        DESCRIPTION.
    dir_data : TYPE
        DESCRIPTION.
    dir_tomo : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    for per in tqdm(Pers):
        f_tph=[]
        f_tgr=[]
        os.system('mkdir -p %s%.2fs_snrcri_%d'%(dir_tomo,per,snrcri))
        
        fHV=glob(dir_tomo+'/../HV/%.1fsec*/RV_%s_*.txt'%(per,event))
        if len(fHV)==1:
            StasHV=np.loadtxt(fHV[0],dtype=str, usecols=1,unpack=True)
        else:
            pass

        #################### linear interpolate measurments (phase travel time / amplitude) to integer periods ##############
        for idx in range(len(Fps)):
            fp=Fps[idx]
            
            if not os.path.isfile(dir_data+'/'+fp+'_1_DISP.0'):
                continue

            perA,vphA,vgrA    = np.loadtxt(dir_data+'/'+fp+'_1_DISP.0',dtype=float,usecols=(2,4,3),unpack=True)
            perA1,vphA1,vgrA1 = np.loadtxt(dir_data+'/'+fp+'_r_1_DISP.0',dtype=float,usecols=(2,4,3),unpack=True)
            
            perSA, snrSA   = np.loadtxt(dir_data+'/'+fp+'_snr_rms.txt',dtype=float,usecols=(0,2),unpack=True)
            perSA1, snrSA1 = np.loadtxt(dir_data+'/'+fp+'_r_snr_rms.txt',dtype=float,usecols=(0,2),unpack=True)
            
            vph  = get_valT(per,perA,vphA,fill_value=np.nan)
            vph1 = get_valT(per,perA1,vphA1,fill_value=np.nan)
            
            vgr  = get_valT(per,perA,vgrA,fill_value=np.nan)
            vgr1 = get_valT(per,perA1,vgrA1,fill_value=np.nan)
            
            snr  = get_valT(per,perSA,snrSA,fill_value=np.nan)
            snr1 = get_valT(per,perSA1,snrSA1,fill_value=np.nan)  
            
            if np.isnan(vph) or np.isnan(vph1): #debug
                print(event,per,fp)
                continue
            if snr<snrcri and snr1<snrcri:
                continue
            if snr<snr1:
                vph=vph1
                vgr=vgr1
            
            ########use with caution
            # tmpstr=fp.split('COR_')[1]
            # if not tmpstr[:3] ==event:
            #     sta=tmpstr[:3]
            # else:
            #     sta=tmpstr.split('_')[1][:3]
            # # print(sta)
            # if not sta in StasHV:
            #     continue
            ########use with caution

            f_tph.append('%f %f %f'%(Stlo[idx],Stla[idx],Dists[idx]/vph))
            f_tgr.append('%f %f %f'%(Stlo[idx],Stla[idx],Dists[idx]/vgr))
            # f_amplitude.append('%f %f %f'%(Stlo[idx],Stla[idx],vph))
        np.savetxt(dir_tomo+'%.2fs_snrcri_%d'%(per,snrcri)+'/tph_%s.txt'%(event),f_tph,fmt='%s')
        np.savetxt(dir_tomo+'%.2fs_snrcri_%d'%(per,snrcri)+'/tgr_%s.txt'%(event),f_tgr,fmt='%s')
        # np.savetxt(dir_tomo+'%ds_snrcri_%d'%(per,snrcri)+'/amplitude_%s.txt'%(event),f_amplitude,fmt='%s')
        
    
    
def correct_2pi_phase_front(per,event,evla,evlo,dir_tomo,snrcri=10,t_type='ph'):
    '''
    Parameters
    ----------
    per : TYPE
        DESCRIPTION.
    event : TYPE
        DESCRIPTION.
    dir_tomo : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    # snrcri=10
    
    # evla=Evlas[0]
    # evlo=Evlos[0]
    # not all stations are used in the future, snr cri...
    if not os.path.isfile(dir_tomo+'%.2fs_snrcri_%d'%(per,snrcri)+'/t%s_%s.txt'%(t_type,event)):
        print('no file, event %s, per %.2fs'%(event,per))
        return
    ########## ! No 2pi correction ##########
    # os.system('cp '+dir_tomo+'%.2fs_snrcri_%d'%(per,snrcri)+'/t%s_%s.txt'%(t_type,event)+' '+dir_tomo+'%.2fs_snrcri_%d'%(per,snrcri)+'/t%s_%s.txt_v1'%(t_type,event))
    # return
    ########## ! No 2pi correction ##########
    try:
        Stlo,Stla,travel_time=np.loadtxt(dir_tomo+'%.2fs_snrcri_%d'%(per,snrcri)+'/t%s_%s.txt'%(t_type,event),unpack=True)
    except:
        print('empty file, %.2fs_snrcri_%d'%(per,snrcri)+'/t%s_%s.txt'%(t_type,event))
        return
    
    travel_time_v1=travel_time.copy() #travel time after 2pi correction
    if not type(Stlo)==np.ndarray:
        return
    
    
    Dists_Ev=np.empty(len(Stlo))
    Dists_Rs=Dists_Ev.copy() #distance to reference station
    for idx in range(len(Stlo)):
        stla=Stla[idx]
        stlo=Stlo[idx]
        Dists_Ev[idx]=get_dist(evla,evlo,stla,stlo)#gps2dist_azimuth(evla,evlo,stla,stlo)[0]/1000 #in km
    argsort_Ev=np.argsort(Dists_Ev) #distance to event sorted index
    
    #Find reference station that is far from source
    idx0=np.nan
    for idxE in argsort_Ev:
        # if Dists_Ev[idxE] < wlcri*per:
        #     continue
        if travel_time[idxE] < nper*per:
            continue
        if travel_time[idxE]<Dists_Ev[idxE]/vmax and travel_time[idxE]>Dists_Ev[idxE]/vmin:
            continue
        # if np.abs(Dists_Ev[idxE]/travel_time[idxE]-np.nanmean(Dists_Ev[:-1]/travel_time[:-1]))>2*np.nanstd(Dists_Ev[:-1]/travel_time[:-1]):
        #     continue
        idx0=idxE #reference station index
        break
    if np.isnan(idx0):
        return
        
    for idx in range(len(Stlo)):
        stla=Stla[idx]
        stlo=Stlo[idx]
        Dists_Rs[idx]=get_dist(Stla[idx0],Stlo[idx0],stla,stlo) #gps2dist_azimuth(Stla[idx0],Stlo[idx0],stla,stlo)[0]/1000 #in km
    argsort_Rs=np.argsort(Dists_Rs) #distance to reference station sorted index
    for i in range(len(argsort_Rs)):
        if i==0:
            ########### whether using a reference phase velocity for initial reference station correction? ###############
            tr_ti=travel_time[argsort_Rs[i]]
            # RefVphDict={'0.1':0.5,'0.2':0.5,'0.3':0.6,'0.4':0.8,'0.5':0.8,
            #             '0.6':0.8,'0.7':0.8,'0.8':0.8,'0.9':0.8,'1.0':0.8}
            RefVphDict={'0.1':0.8,'0.2':0.8,'0.3':0.8,'0.4':0.8,'0.5':0.8,
                        '0.6':0.8,'0.7':0.8,'0.8':0.8,'0.9':0.8,'1.0':0.8}
            tmp_RefVph=RefVphDict['%.1f'%per] #0.5km/s at 0.2s, 0.6km/s at 0.3s, 0.8km/s at 0.5s and 0.4s      #for 0.2 s, 5 Hz  #0.8 #0.8 # 0.8km/s reference phase velocity
            tr_Rs_pred1=Dists_Ev[argsort_Rs[i]]/tmp_RefVph
            N1_trRs= np.ceil((tr_ti-tr_Rs_pred1)/per-1/2) #number of cycles to correct, predicted travel time +-per/2
            tr_ti_v1=tr_ti-N1_trRs*per
            travel_time_v1[argsort_Rs[i]]=tr_ti_v1
            ########### whether using a reference phase velocity for initial reference station correction? ###############

            # travel_time_v1[argsort_Rs[i]]=travel_time[argsort_Rs[i]]
            
            continue
        j=0
        # dist_ij=gps2dist_azimuth(Stla[argsort_Rs[i]],Stlo[argsort_Rs[i]],Stla[argsort_Rs[j]],Stlo[argsort_Rs[j]])[0]/1000
        dist_ij=get_dist(Stla[argsort_Rs[i]],Stlo[argsort_Rs[i]],Stla[argsort_Rs[j]],Stlo[argsort_Rs[j]])

        for tmp_j in range(i):
            # if Dists_Ev[argsort_Rs[tmp_j]]<wlcri*per: #1e-9:
            #     continue
            if travel_time[argsort_Rs[tmp_j]]<nper*per:
                continue
            dist_tmp=get_dist(Stla[argsort_Rs[i]],Stlo[argsort_Rs[i]],Stla[argsort_Rs[tmp_j]],Stlo[argsort_Rs[tmp_j]]) #gps2dist_azimuth(Stla[argsort_Rs[i]],Stlo[argsort_Rs[i]],Stla[argsort_Rs[tmp_j]],Stlo[argsort_Rs[tmp_j]])[0]/1000
            if dist_tmp<dist_ij:
                j=tmp_j
                dist_ij=dist_tmp
        tr_ti=travel_time[argsort_Rs[i]]
        
        tr_ti_pred0=travel_time[argsort_Rs[j]]/Dists_Ev[argsort_Rs[j]]*Dists_Ev[argsort_Rs[i]]
        tr_ti_pred1=travel_time_v1[argsort_Rs[j]]/Dists_Ev[argsort_Rs[j]]*Dists_Ev[argsort_Rs[i]]
        
        N0_tr= np.ceil((tr_ti-tr_ti_pred0)/per-1/2) #number of cycles to correct, predicted travel time +-per/2
        N1_tr= np.ceil((tr_ti-tr_ti_pred1)/per-1/2) #number of cycles to correct, predicted travel time +-per/2
        
        ##########################################
        # if N0_tr==0 or N1_tr==0:
        #     tr_ti_v1=tr_ti
        # else:
        #     tr_ti_v1=tr_ti-N0_tr*per
        tr_ti_v1=tr_ti-N1_tr*per
        ##########################################
        # if N_tr!=0:
        #     print(N_tr,tr_ti,tr_ti_0,travel_time[argsort_Rs[j]],Dists_Ev[argsort_Rs[j]],Dists_Ev[argsort_Rs[i]])
        
        travel_time_v1[argsort_Rs[i]]=tr_ti_v1
        # if Stla[argsort_Rs[i]]==36.3391:
        #     break
    f_travel_time_v1=[]
    for idx in range(len(Stlo)):
        f_travel_time_v1.append('%f %f %f'%(Stlo[idx],Stla[idx],travel_time_v1[idx]))
    np.savetxt(dir_tomo+'%.2fs_snrcri_%d'%(per,snrcri)+'/t%s_%s.txt_v1'%(t_type,event),f_travel_time_v1,fmt='%s')
    return Stla[idx0],Stlo[idx0]
    
def Interp_Min_Curv(per,fp,T,dir_tomo,snr=10,Ncpu=40):
    #-I0.001
    if not os.path.isfile(dir_tomo+'Scripts/C_plot_travel'):
        #-I0.0002
        Cpt=['#!/bin/csh','if ($#argv != 2) then','  echo "USAGE: C_plot_travel [input file] [T]"','  exit 1','endif',
             'set input_map = $argv[1]',
             'set T = ${argv[2]}',
             'set REG = -R247.49/247.53/38.84/38.875\n',
             
             'gmtset BASEMAP_TYPE fancy',
             'surface $input_map -T$T -G$input_map"".grd -I0.0005 $REG',
             'grd2xyz $input_map"".grd $REG > $input_map".HD_$T"',
             'rm $input_map"".grd'] #C_plot_travel
        os.chdir(dir_tomo)
        os.system('mkdir -p Scripts')
        np.savetxt(dir_tomo+'Scripts/C_plot_travel',Cpt,fmt='%s')
        os.system('chmod 755 Scripts/C_plot_travel')
        
    os.chdir(dir_tomo+'%.2fs_snrcri_%d'%(per,snrcri))
    # HD=['ml gcc/8.5.0 intel-oneapi-mpi/2021.4.0 gmt/6.2.0']
    HD=[]
    if type(fp)==str:
        HD.append('../Scripts/C_plot_travel '+fp+' '+str(T))
    else:
        n=0
        for tmpfp in fp:
            HD.append('../Scripts/C_plot_travel '+tmpfp+' '+str(T)+' &')
            n+=1
            if n==Ncpu:
                HD.append('wait')
                n=n-Ncpu
        HD.append('wait')
    np.savetxt('HD_T%s.csh'%(str(T)),HD,fmt='%s')
    os.system('csh HD_T%s.csh'%(str(T)))

def correct_curvature_sta(per,event,dir_tomo,T=0,snrcir=10,t_type='ph',output=False):
    try:
        stlo,stla,travel_time=np.loadtxt(dir_tomo+'%.2fs_snrcri_%d'%(per,snrcri)+'/t%s_%s.txt_v1'%(t_type,event),unpack=True)
        grlo,grla,travel_timeHD=np.loadtxt(dir_tomo+'%.2fs_snrcri_%d'%(per,snrcri)+'/t%s_%s.txt_v1.HD_%s'%(t_type,event,str(T)),unpack=True)   
    except:
        print('empty file: %f, %s'%(per,event))
        return []
    if not isinstance(stlo,np.ndarray):
        print('wrong')
        return []
    # dx=111*0.2
    # dy=111*0.2
    
    
    X,ind=np.unique(grlo,return_index=True);X=X[np.argsort(ind)] #original index
    Y,ind=np.unique(grla,return_index=True);Y=Y[np.argsort(ind)]
    XX=grlo.reshape((len(Y),len(X)))
    YY=grla.reshape((len(Y),len(X)))
    ZZ=travel_timeHD.reshape((len(Y),len(X)))
    
    d_deg=0.0005 #d_deg=0.0002 #TBD 0.001
    dy0=get_dist(np.min(Y),np.mean(X),np.max(Y),np.mean(X))/len(Y)
    dx0=get_dist(np.mean(Y),np.min(X),np.mean(Y),np.max(X))/len(X)
    r_yx=dy0/dx0 #ratio of dy over dx <= dy/dx  
    
    dZ2=[]
    fp1=[]
    for k in range(len(stlo)):
        # if travel_time[k]<nper*per:
        #     fp1.append('%f %f %f'%(stlo[k],stla[k],travel_time[k]))
        #     continue
        tmp_dist=9e9
        tmp_j=0
        tmp_i=0
        for j in range(2,len(Y)-2):
            if np.abs(YY[j,0]-stla[k])>d_deg: #YY same value in second axis
                continue
            for i in range(2,len(X)-2):
                if np.abs(XX[0,i]-stlo[k])>d_deg: # XX same value in first axis
                    continue
                
                dist0=get_dist(stla[k],stlo[k],YY[j,i],XX[j,i])
                if dist0<tmp_dist:

                    tmp_dist=dist0
                    tmp_j=j
                    tmp_i=i
                
        j=tmp_j
        i=tmp_i
        
        dx=get_dist(YY[j,i+1],XX[j,i+1],YY[j,i-1],XX[j,i-1])/2
        dy=get_dist(YY[j+1,i],XX[j+1,i],YY[j-1,i],XX[j-1,i])/2
        # Radius=6371.1391285
        # dx=d_deg*Radius*np.pi/180
        # dy=d_deg*Radius*np.pi/180
        
        dZ2_dx2=(-ZZ[j,i+2]+16*ZZ[j,i+1]-30*ZZ[j,i]+16*ZZ[j,i-1]-ZZ[j,i-2])/12/(dx**2)
        dZ2_dy2=(-ZZ[j+2,i]+16*ZZ[j+1,i]-30*ZZ[j,i]+16*ZZ[j-1,i]-ZZ[j-2,i])/12/(dy**2)
        # dZ2.append(np.abs(dZ2_dx2+dZ2_dy2))
        dZ2.append(dZ2_dx2+dZ2_dy2)
        #################################!!! curvature criteria !!!####################################
    for k in range(len(stlo)):
        if travel_time[k]<nper*per:
            fp1.append('%f %f %f'%(stlo[k],stla[k],travel_time[k]))
            continue
        if np.abs((dZ2[k]-np.mean(dZ2)))>np.std(dZ2): #bigger than 1 std
            # print(j,i)
            continue
        # if dZ2[k]>
        fp1.append('%f %f %f'%(stlo[k],stla[k],travel_time[k]))
    
    np.savetxt(dir_tomo+'%.2fs_snrcri_%d'%(per,snrcri)+'/t%s_%s.txt_v2'%(t_type,event),fp1,fmt='%s')
    if output:
        return dZ2
    else:
        return

def Travel2Slow(per,event,dir_tomo,t_type='ph',cri_quad_sta=0.25,snrcri=10):
    ###############################
    try:
        stlo,stla,junk=np.loadtxt(dir_tomo+'%.2fs_snrcri_%d'%(per,snrcri)+'/t%s_%s.txt_v2'%(t_type,event),unpack=True)
        stloHD,stlaHD,travel_timeHD=np.loadtxt(dir_tomo+'%.2fs_snrcri_%d'%(per,snrcri)+'/t%s_%s.txt_v2.HD_0'%(t_type,event),unpack=True)
        stloHD,stlaHD,travel_timeHD1=np.loadtxt(dir_tomo+'%.2fs_snrcri_%d'%(per,snrcri)+'/t%s_%s.txt_v2.HD_0.2'%(t_type,event),unpack=True)
    except:
        print('cannot open file: %.2fs, %s'%(per,event))
        return
    # X=np.unique(stloHD)
    # Y=np.unique(stlaHD)
    X,ind=np.unique(stloHD,return_index=True);X=X[np.argsort(ind)] #original index
    Y,ind=np.unique(stlaHD,return_index=True);Y=Y[np.argsort(ind)]
    d_deg=0.0005#d_deg=0.0002 #0.001
    dy0=get_dist(np.min(Y),np.mean(X),np.max(Y),np.mean(X))/len(Y)
    dx0=get_dist(np.mean(Y),np.min(X),np.mean(Y),np.max(X))/len(X)
    r_yx=dy0/dx0 #ratio of dy over dx <= dy/dx  
        
    XX=stloHD.reshape((len(Y),len(X)))
    YY=stlaHD.reshape((len(Y),len(X)))
    ZZ=travel_timeHD.reshape((len(Y),len(X)))
    ZZ1=travel_timeHD1.reshape((len(Y),len(X)))
    
    Slow=np.zeros(ZZ.shape)
    Azi=np.zeros(ZZ.shape)
    slow_r=0
    azi_r=999
    
    slow_azi=[]
    for i in range(0,ZZ.shape[1]):
        for j in range(0,ZZ.shape[0]):
            if i<2 or i>ZZ.shape[1]-3 or j<2 or j>ZZ.shape[0]-3:
                ZZ[j,i]=0
                continue
            if ZZ[j,i]<nper*per: #wavelength criteria
                ZZ[j,i]=0
                continue
            if np.abs(ZZ[j,i]-ZZ1[j,i])>per/4:
                ZZ[j,i]=0
                continue
            
            Quadrant=np.zeros(4)
            
            cri_close=np.logical_and.reduce((np.abs(stlo-XX[j,i])<d_deg/dx0*cri_quad_sta,
                                            np.abs(stla-YY[j,i])<d_deg/dy0*cri_quad_sta))
            if np.sum(cri_close)<3:
                ZZ[j,i]=0
                continue
            stlo1=stlo[cri_close]
            stla1=stla[cri_close]
            
            for k in range(len(stlo1)):
                # if np.abs(stlo[k]-XX[j,i])>d_deg/dx0*cri_quad_sta:
                #     n+=1
                #     continue
                # if np.abs(stla[k]-YY[j,i])>d_deg/dy0*cri_quad_sta:
                #     n+=1
                #     continue
                dist0=get_dist(stla1[k],stlo1[k],YY[j,i],XX[j,i])
                # dist0=1
                if dist0>cri_quad_sta:
                    continue
                if stla1[k]>+YY[j,i]:
                    if stlo1[k]>=XX[j,i]:
                        Quadrant[0]+=1
                    else:
                        Quadrant[1]+=1
                else:
                    if stlo1[k]<=XX[j,i]:
                        Quadrant[2]+=1
                    else:
                        Quadrant[3]+=1
            if np.sum(Quadrant>0)<3:
                ZZ[j,i]=0
                continue
            # print(n)
    for j in range(0,ZZ.shape[0]):
        for i in range(0,ZZ.shape[1]):
    # for i in range(0,ZZ.shape[1]):
    #     for j in range(0,ZZ.shape[0]):
            if i<2 or i>ZZ.shape[1]-3 or j<2 or j>ZZ.shape[0]-3:
                slow_azi.append('%f %f %f %f'%(XX[j,i],YY[j,i],slow_r,azi_r))
                continue
            dx=get_dist(YY[j,i+1],XX[j,i+1],YY[j,i-1],XX[j,i-1])/2
            dy=get_dist(YY[j+1,i],XX[j+1,i],YY[j-1,i],XX[j-1,i])/2
            dZ_dx=(ZZ[j,i+1]-ZZ[j,i-1])/dx/2
            dZ_dy=(ZZ[j+1,i]-ZZ[j-1,i])/dy/2
            
            # dZ2_dx2=(-ZZ[j,i+2]+16*ZZ[j,i+1]-30*ZZ[j,i]+16*ZZ[j,i-1]-ZZ[j,i-2])/12/(dx**2)
            # dZ2_dy2=(-ZZ[j+2,i]+16*ZZ[j+1,i]-30*ZZ[j,i]+16*ZZ[j-1,i]-ZZ[j-2,i])/12/(dy**2)
            # if np.abs(dZ2_dx2+dZ2_dy2)>0.005:
            #     slow_azi.append('%f %f %f %f'%(XX[j,i],YY[j,i],slow_r,azi_r))
            #     continue

            Slow[j,i]=np.sqrt(dZ_dx**2+dZ_dy**2)
            Azi[j,i]=np.arctan2(dZ_dy,dZ_dx)/np.pi*180
            if Slow[j,i]>1/vmin or Slow[j,i]<1/vmax or ZZ[j,i+1]==0 or ZZ[j,i-1]==0 or ZZ[j+1,i]==0 or ZZ[j-1,i]==0:
                Slow[j,i]=0
                Azi[j,i]=999
            
            slow_azi.append('%f %f %f %f'%(XX[j,i],YY[j,i],Slow[j,i],Azi[j,i]))
    
    np.savetxt(dir_tomo+'%.2fs_snrcri_%d'%(per,snrcri)+'/slow_azi_%s_%s.txt_v2.HD'%(t_type,event),slow_azi,fmt='%s')
    # np.savetxt(dir_tomo+'%.2fs_snrcri_%d'%(per,snrcri)+'/slow_azi_%s.txt_v1.HD_0.2'%(event),slow_azi,fmt='%s')

def slow_map_to_iso_map(dir_tomo,per,t_type='ph',nsource=10,snrcri=10):
    
    Fps=glob(dir_tomo+'%.2fs_snrcri_%d'%(per,snrcri)+'/slow_azi_%s_*.txt_v2.HD'%t_type)
    stloHD,stlaHD,slowHD, aziHD = np.loadtxt(Fps[0],unpack=True)
    # stloHD,stlaHD,travel_timeHD=np.loadtxt(dir_tomo+'%ds_snrcri_%d'%(per,snrcri)+'/travel_time_%s.txt_v1.HD_0'%('Z58A'),unpack=True)
    X,ind=np.unique(stloHD,return_index=True);X=X[np.argsort(ind)] #original index
    Y,ind=np.unique(stlaHD,return_index=True);Y=Y[np.argsort(ind)]
    
    # tmp=X
    # X=Y
    # Y=tmp
        
    XX=stloHD.reshape((len(Y),len(X)))
    YY=stlaHD.reshape((len(Y),len(X)))
    SLOW=np.empty(shape=(len(Y),len(X))+(0,)).tolist()
    AZI=np.empty(shape=(len(Y),len(X))+(0,)).tolist()
    # read slowness measurements from each source (earthquake source or ambient noise source station)
    for fp in Fps:
        stloHD,stlaHD,slowHD, aziHD = np.loadtxt(fp,unpack=True)
        ZZ=slowHD.reshape((len(Y),len(X)))
        ZZ1=aziHD.reshape((len(Y),len(X)))
        for i in range(0,ZZ.shape[1]):
            for j in range(0,ZZ.shape[0]):
                if ZZ[j,i]<1/vmax or ZZ[j,i]>1/vmin:
                    continue
                else:
                    SLOW[j][i].append(ZZ[j,i])
                    AZI[j][i].append(ZZ1[j,i])
    # calculate isotropic phase velocity and uncertainty (weighted by azimuthal distribution)
    slow_iso=np.zeros(XX.shape)
    vph_un=np.zeros(XX.shape);
    
    flag=0
    fiso=[]
    for j in range(0,ZZ.shape[0]):
        for i in range(0,ZZ.shape[1]):
            if len(SLOW[j][i])>nsource:
                slow=SLOW[j][i]
                count=np.zeros(len(slow))
                weight=np.zeros(len(slow))
                
                tmpazi=np.array(AZI[j][i])
                
                for k in range(len(slow)):
                    count[k]=np.sum( np.logical_or.reduce((np.abs(tmpazi[k]-tmpazi)<25,np.abs(tmpazi[k]-tmpazi)>335)) )
                    # count[k]=np.sum( np.abs(tmpazi[k]-tmpazi)<25 )
                weight=1/count
                weight=weight/np.sum(weight)
                
                # weight=np.ones(len(slow))/len(slow)
                
                w2=np.sum(weight**2)
                s0=np.sum(weight*slow)
                slow_iso[j,i]=s0
                vph_un[j,i]= np.sqrt( np.sum(weight*((slow-np.mean(slow))**2))*w2/(1-w2))/ (s0**2) #np.sqrt(np.sum((slow-np.mean(slow))**2)/len(slow)/(len(slow)-1))/s0**2#
                
                fiso.append('%.6f %.6f %.6f %.6f %d'%(XX[j,i],YY[j,i],1/s0,vph_un[j,i],len(SLOW[j][i])))
            else:
                fiso.append('%.6f %.6f 0 999 0'%(XX[j,i],YY[j,i]))
    
    np.savetxt(dir_tomo+'%.2fs_snrcri_%d'%(per,snrcri)+'/%.2fs_%s_iso_ani_v1_scale_0.iso'%(per,t_type),fiso,fmt='%s')

#%%
# if __name__ == '__main__':
snrcri=10 #? whether used
wlcri=1.2 #8 !!! could be optimized? only use travel time info. for selection, not relying on the pre-assumed slowness
vmin=0.1
vmax=1.5
# vmin=0.1
# vmax=5 ##########
nper=1.0 # wavelength cri, nper wavelengths
NCPU=30

# dir_data='/uufs/chpc.utah.edu/common/home/flin-group4/qicheng/Meadow/SAC_v1/stack_all/STACK_ZNE/'
# dir_tomo='/uufs/chpc.utah.edu/common/home/flin-group4/qicheng/Meadow/SAC_v1/stack_all/STACK_ZNE/Tomo/'
dir_data='/uufs/chpc.utah.edu/common/home/flin-group4/qicheng/Meadow/SAC_v2/stack_all_Drift/STACK_ZNE/'
dir_tomo='/uufs/chpc.utah.edu/common/home/flin-group4/qicheng/Meadow/SAC_v2/stack_all_Drift/STACK_ZNE/Tomo_p005/'
dir_tomo1=dir_tomo
# dir_data='/uufs/chpc.utah.edu/common/home/flin-group5/qicheng/ANT_Adj/'
# dir_tomo='/uufs/chpc.utah.edu/common/home/flin-group5/qicheng/ANT_Adj/Tomo_Wells1D/'
# dir_Adj='/uufs/chpc.utah.edu/common/home/flin-group5/qicheng/ANT_Adj/'

#%%
# event='Wells_PREM'
event='001'
# PersAN=np.arange(8,24.1,2)
# PersEQ=np.concatenate(([24],np.arange(30,100.1,10)))
# PersWUS=np.unique(np.concatenate((PersAN, PersEQ)))
# Pers=np.array([0.1,0.2,0.5,1,2,5,10])#PersWUS.copy()
# Pers=np.array([0.2])#PersWUS.copy()
Pers=np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
# Pers=np.array([0.5])
# Pers=np.array([0.5])

# Pers=[60]
#%%

dir_events=dir_data
# Events0=np.loadtxt(dir_events+'Events0.lst',dtype=str,usecols=0,unpack=True)

# Events,Evt0s=np.loadtxt(dir_events+'Events.lst',dtype=str,usecols=(0,4),unpack=True)
Events=np.loadtxt(dir_events+'station_line.lst',dtype=str,usecols=0,unpack=True)
Evlas,Evlos=np.loadtxt(dir_events+'station_line.lst',usecols=(2,3),unpack=True)

Stnms=np.loadtxt(dir_events+'station_line.lst',dtype=str,usecols=0,unpack=True)
Stlas,Stlos=np.loadtxt(dir_events+'station_line.lst',dtype=float,usecols=(2,3),unpack=True)

plt.figure(figsize=(8,8))
plt.plot(Stlos,Stlas,'k^')
plt.plot(Evlos,Evlas,'r*',markersize=10)
# plt.xlim([-132,-99])
# plt.ylim([27,54])
plt.grid()
# plt.savefig(dir_savefig+'StaEv.png')
#%%
# dir_tomo1='/uufs/chpc.utah.edu/common/home/flin-group4/qicheng/Meadow/SAC_v1/stack_all/STACK_ZNE/Tomo_scal/'

# dir_tomo='/uufs/chpc.utah.edu/common/home/flin-group4/qicheng/Meadow/SAC_v2/stack_all_Drift/STACK_ZNE/Tomo/'

# per=0.5
# t_type='ph'
# v_type='_v1'

# n=0
# for iev in range(len(Events)):
#     try:
#         ev=Events[iev]
#         lon,lat,tr_scal=np.loadtxt(dir_tomo1+'%.2fs_snrcri_%d'%(per,snrcri)+'/t%s_%s.txt%s'%(t_type,ev,v_type),dtype=float,unpack=True)
#         lon,lat,tr=np.loadtxt(dir_tomo+'%.2fs_snrcri_%d'%(per,snrcri)+'/t%s_%s.txt%s'%(t_type,ev,v_type),dtype=float,unpack=True)
#         if np.abs(np.median(tr_scal-tr))>0.1:
#             print(ev,np.mean(tr_scal-tr),np.median(tr_scal-tr))
#             n+=1
#         # if ev=='069'
        
#     except:
#         continue

#%%
# plt.close('all')
# dir_tomo1='/uufs/chpc.utah.edu/common/home/flin-group4/qicheng/Meadow/SAC_v1/stack_all/STACK_ZNE/Tomo/'
per=0.5
t_type='ph'
v_type='_v2'  #'_v1'
tmp_sta='001'
# iev=4
# iev=np.where(Events=='202003312352')[0][0] #202002130816
# iev=np.where(Events=='201007072353')[0][0]
# iev=np.where(Events=='201904222144')[0][0]
iev=np.where(Events==tmp_sta)[0][0] #'130' #'011'
ev=Events[iev]
event=ev
lon,lat,tr=np.loadtxt(dir_tomo1+'%.2fs_snrcri_%d'%(per,snrcri)+'/t%s_%s.txt%s'%(t_type,ev,v_type),dtype=float,unpack=True)
VMIN=0
VMAX=10#np.median(tr)*2
plt.figure(figsize=(12,6))
plt.scatter(lon,lat,c=tr,cmap=cmnew,vmin=VMIN,vmax=VMAX,s=132)
plt.plot(Evlos[iev]+360,Evlas[iev],'r*')
plt.colorbar(format='%.2f',ticks=np.linspace(VMIN,VMAX,7)) #orientation='horizontal', 

dir_savefig='/uufs/chpc.utah.edu/common/home/u1318104/Figures/Zanskar_Meadow/05312024_VphEikon/'
# plt.savefig(dir_savefig+'sta%s_%.1fs%s.png'%(tmp_sta,per,v_type))
# longr,latgr,trgr=np.loadtxt(dir_tomo+'%.2fs_snrcri_%d'%(per,snrcri)+'/t%s_%s.txt%s.HD_0'%(t_type,ev,v_type),dtype=float,unpack=True)
# VMIN=0
# VMAX=np.median(tr)*2
# plt.figure(figsize=(12,6))
# plt.scatter(longr,latgr,c=trgr,cmap=cmnew,vmin=VMIN,vmax=VMAX,s=2)
# plt.scatter(lon,lat,c=tr,cmap=cmnew,vmin=VMIN,vmax=VMAX,s=4,marker='^')
# plt.plot(Evlos[iev]+360,Evlas[iev],'r*')
# plt.colorbar(format='%.2f',ticks=np.linspace(VMIN,VMAX,7)) #orientation='horizontal', 
# # plt.xlim([230,260])
# # plt.ylim([28,50])
#%%
dir_tomo1='/uufs/chpc.utah.edu/common/home/flin-group4/qicheng/Meadow/SAC_v1/stack_all/STACK_ZNE/Tomo_junk/'
ev='001'
iev=np.where(Events==ev)[0][0]
longr,latgr,slowgr,azigr=np.loadtxt(dir_tomo+'%.2fs_snrcri_%d'%(per,snrcri)+'/slow_azi_%s_%s.txt_v2.HD'%(t_type,ev),dtype=float,unpack=True)
stlo,stla,travel_time=np.loadtxt(dir_tomo+'%.2fs_snrcri_%d'%(per,snrcri)+'/t%s_%s.txt_v2'%(t_type,ev),unpack=True)
# VMIN=np.nanmin(1/slowgr)
# VMAX=np.nanmax(1/slowgr[slowgr!=0])
# VMIN=np.mean(1/slowgr[slowgr!=0])-0.5
# VMAX=np.mean(1/slowgr[slowgr!=0])+0.5
VMIN=np.max(1/slowgr[slowgr!=0])
VMAX=np.min(1/slowgr[slowgr!=0])
# VMIN=0.24 #0.09
# VMAX=0.745#1.09
# VMIN=3.68
# VMAX=4.16
plt.figure(figsize=(7,7))
plt.scatter(stlo,stla,c='k',s=30)

plt.scatter(longr,latgr,c=1/slowgr,cmap=cmnew,vmin=VMIN,vmax=VMAX,s=20)
# plt.scatter(lon,lat,c=tr,cmap=cmnew,vmin=VMIN,vmax=VMAX,s=4,marker='^')
plt.plot(Evlos[iev]+360,Evlas[iev],'r*')
plt.colorbar(format='%.2f',ticks=np.linspace(VMIN,VMAX,7),pad=0.08,fraction=0.046,shrink=0.5,orientation='horizontal') #orientation='horizontal', 

plt.title('%.2fs'%per)
# plt.xlim([230,260]) 
# plt.ylim([28,52])
plt.xlim([-112.51+360,-112.47+360])
plt.ylim([38.84,38.875])
# xmax=-112.47+360 #-100+360
# ymax=38.875 #52
# xmin=-112.51+360 #-130
# ymin=38.84 #28


plt.grid()
dir_savefig='/uufs/chpc.utah.edu/common/home/u1318104/Figures/Zanskar_Meadow/06212024_Beam/'
# plt.savefig(dir_savefig+'%s_%ds.png'%(ev,per))
#%% get good measurements - linear interpolation for integer periods
# os.chdir(dir_data)
# os.system('mkdir -p Ev_fp')
event='107'
# Para_Pool=[]
# for event in tqdm(Events):
#     # os.chdir(dir_data+event)
#     if True:#not os.path.isfile('tmp_station.lst'):
#         np.savetxt('get_stationinfo.csh',['saclst evla evlo stla stlo dist f ???/COR_*%s*.SAC_ZZ > Ev_fp/Ev%s.lst'%(event,event)],fmt='%s')
#         os.system('csh get_stationinfo.csh')

#%%
Para_Pool=[]

# Events=np.array(['001'])
# Pers=np.array([0.5])
for event in tqdm(Events):
    Fps=np.loadtxt(dir_data+'/Ev_fp/Ev%s.lst'%event,usecols=(0),dtype=str,unpack=True)
    Evla,Evlo,Stla,Stlo,Dists=np.loadtxt(dir_data+'/Ev_fp/Ev%s.lst'%event,usecols=(1,2,3,4,5),dtype=float,unpack=True)
    for ifp in range(len(Fps)):
        if Fps[ifp].split('.')[0].split('_')[-1]==event:
            Stla[ifp]=Evla[ifp]
            Stlo[ifp]=Evlo[ifp]
            
    
    if np.mean(Evlo)<0:
        Evlo=Evlo+360
    if np.mean(Stlo)<0:
        Stlo=Stlo+360
    Para_Pool.append((Pers,event,Fps,Stlo,Stla,Dists,dir_data,dir_tomo))
    # break
#%%
pool=Pool(NCPU)
results=pool.starmap(get_good_measurements,Para_Pool)
pool.close()
pool.join()
print('process finished')    




#%% 2pi corrections, with OR without curvature correction
# Pers=np.array([0.2])
Para_Pool=[]
for per in Pers:
    for i in range(len(Events)):
        
        event=Events[i]
        evla=Evlas[i]
        evlo=Evlos[i]
        # if not event=='001':
        #     continue
        # correct_2pi_phase_front(per,event,evla,evlo,dir_tomo,snrcri=5,t_type='ph')
        
        # Para_Pool.append((per,event,evla,evlo,dir_tomo))
        Para_Pool.append((per,event,evla,evlo,dir_tomo,10,'ph'))

    
pool=Pool(NCPU)
results=pool.starmap(correct_2pi_phase_front,Para_Pool)
pool.close()
pool.join()
print('process finished')
#%% initial grid surface
# for per in Pers:
#     fp='tgr_%s.txt'%event
#     Interp_Min_Curv(per,fp,0,dir_tomo,snr=5,Ncpu=10)
# Pers=np.array([0.5])
########################################################################
for per in tqdm(Pers):
    fp=[]
    for event in Events:
        # if not event=='001':
        #     continue
        # fp.append('travel_time_%s.txt'%event)
        # fp.append('tph_%s.txt_v1'%event)
        fp.append('tph_%s.txt_v1'%event)

    
    Interp_Min_Curv(per,fp,0,dir_tomo,snr=10,Ncpu=NCPU)
#%% correct curvature, remove spurious stations with big curvature
for per in Pers:
    for ev in Events:
        correct_curvature_sta(per,ev,dir_tomo,t_type='ph',output=True)


#%%
# per=0.5
# DZ2=[]
# for ev in Events:
#     # if not ev=='001':
#     #     continue
#     dZ2=correct_curvature_sta(per,ev,dir_tomo,output=True)
#     if len(dZ2):
#         DZ2.append(dZ2)
#         # DZ2+=dZ2
#     # break
# #%%
# i=1
# dZ2=DZ2[i]
# plt.figure()
# plt.hist(np.array(dZ2),bins=40) #bins=10,range=(0,1500) 
# plt.axvline(x=np.mean(dZ2)-np.std(dZ2),color='k')
# plt.axvline(x=np.mean(dZ2)+np.std(dZ2),color='k')

#%%
########################### correct curvature

# for per in Pers:
#     correct_curvature_sta(per,event,dir_tomo,T=0)

###########################################################
# Para_Pool=[]
# for per in Pers:
#     for i in range(len(Events)):
#         event=Events[i]
#         # evla=Evlas[i]
#         # evlo=Evlos[i]
#         Para_Pool.append((per,event,dir_tomo))
    
# pool=Pool(NCPU)
# results=pool.starmap(correct_curvature_sta,Para_Pool)
# pool.close()
# pool.join()
# print('process finished')
#%% surface grid after removal of spurious stations
# for per in Pers:
#     fp='tgr_%s.txt_v1'%event
#     Interp_Min_Curv(per,fp,0,dir_tomo,snr=5,Ncpu=10)
#     Interp_Min_Curv(per,fp,0.2,dir_tomo,snr=5,Ncpu=10)

##################################################################################
fp=[]

for event in Events:
    # if not event=='001':
    #     continue
    # fp.append('tph_%s.txt_v2'%event)
    fp.append('tph_%s.txt_v2'%event)

for per in tqdm(Pers):
    Interp_Min_Curv(per,fp,0,dir_tomo,snr=10,Ncpu=NCPU)
    Interp_Min_Curv(per,fp,0.2,dir_tomo,snr=10,Ncpu=NCPU)




#%%
# for per in Pers:
#     Travel2Slow(per,event,dir_tomo,cri_quad_sta=10,snrcri=5)

# Pers=np.array([0.5])
# Pers=np.array([0.2])
# Pers=np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])

########################################################################
Para_Pool=[]
for per in Pers:
    for i in range(len(Events)):
        event=Events[i]
        # evla=Evla[i]
        # evlo=Evlo[i]
        # if not event=='001':
        #     continue
        Para_Pool.append((per,event,dir_tomo))
    
pool=Pool(NCPU)
results=pool.starmap(Travel2Slow,Para_Pool)
pool.close()
pool.join()
print('process finished')


#%%
for per in tqdm(Pers):
    slow_map_to_iso_map(dir_tomo,per,nsource=10,snrcri=10)

#%%
plt.rcParams.update({
    "text.usetex": True,
    "font.family": 'Times New Roman',
    "font.size": 11,
    "figure.autolayout": True}) #'axes.linewidth': 0.8
#%%
# plt.close('all')
# Pers=np.array([0.1])
# per=0.5
# Pers=np.array([0.5])
for per in Pers:
    # VMINs=[0.248,0.270,0.270,0.207,.2,  .2,.2,.21,.21,.229]
    # VMAXs=[0.356,0.361,0.361,0.319,.316, .316,.316, .384,.384,.391]
    # iper=np.where(per==Pers)[0][0]
    # VMIN=VMINs[iper]
    # VMAX=VMAXs[iper]
    
    t_type='ph'
    stloHD,stlaHD,vgrHD, unHD, nsHD = np.loadtxt(dir_tomo+'%.2fs_snrcri_%d'%(per,snrcri)+'/%.2fs_%s_iso_ani_v1_scale_0.iso'%(per,t_type),unpack=True)
    MA=np.zeros(vgrHD.shape)
    MA[np.where(vgrHD==0)]=1
    vgrHD_ma=np.ma.array(vgrHD,mask=MA)
    # unHD_ma=np.ma.array(unHD,mask=MA)
    
    # vgrHD_ma=np.ma.array(unHD,mask=MA)
    
    VMIN=np.min(vgrHD_ma)
    # VMAX=(np.median(vgrHD_ma)-VMIN)+np.median(vgrHD_ma) #np.max(vgrHD_ma)#
    VMAX=(np.ma.median(vgrHD_ma)-VMIN)+np.ma.median(vgrHD_ma) #np.max(vgrHD_ma)#
    # VMAX=np.max(vgrHD_ma) #np.max(vgrHD_ma)#1

    # vgrHD=unHD
    # MA=np.zeros(vgrHD.shape)
    # MA[np.where(vgrHD==999)]=1
    # vgrHD_ma=np.ma.array(vgrHD,mask=MA)
    # unHD_ma=np.ma.array(unHD,mask=MA)
    
    # VMIN=np.min(vgrHD_ma)
    # # VMAX=(np.median(vgrHD_ma)-VMIN)+np.median(vgrHD_ma) #np.max(vgrHD_ma)#
    # # VMAX=(np.ma.median(vgrHD_ma)-VMIN)+np.ma.median(vgrHD_ma) #np.max(vgrHD_ma)#
    # VMAX=np.max(vgrHD_ma)
    
    # VMIN=0.314
    # VMAX=0.589
    
    # VMAX=np.min(vgrHD_ma)*1.5
    # VMIN=3.68
    # VMAX=4.16
    
    
    #%
    # VMIN=np.min(1/slow_iso)
    # VMAX=np.max(1/slow_iso[slow_iso>0])
    # VMIN=3.5
    # VMAX=4.2
    
    # VMAX=0.5
    
    # plt.figure(figsize=(12,6))
    # plt.scatter(Stlos+360,Stlas,s=10,c='k',marker='^')
    # plt.scatter(stloHD,stlaHD,c=vgrHD_ma,cmap=cmnew,vmin=VMIN,vmax=VMAX,s=4)
    
    
    # plt.colorbar(format='%.2f',ticks=np.linspace(VMIN,VMAX,7)) #orientation='horizontal', 
    
    # VMIN=np.nanmin(unHD_ma)
    # VMAX=np.nanmax(unHD_ma)
    # plt.figure(figsize=(12,6))
    # plt.scatter(stloHD,stlaHD,c=unHD,cmap=cmnew,vmin=VMIN,vmax=VMAX,s=2)
    # plt.colorbar(format='%.2f',ticks=np.linspace(VMIN,VMAX,7)) #orientation='horizontal', 
    #%
    # dir_WUS='/uufs/chpc.utah.edu/common/home/flin-group5/qicheng/ANT_Adj/'
    # fpWUS=np.genfromtxt(dir_WUS+'WUS256.csv',delimiter='|',skip_header=139,usecols=(0,1,2,12,3,7),
    #                   names=['Lat','Lon','Dep','Rho','Vs','Vp'])
    # LatWUS=np.unique(fpWUS['Lat'])
    # LonWUS=np.unique(fpWUS['Lon'])
    # DepWUS=np.unique(fpWUS['Dep'])
    
    LatM=np.unique(stlaHD)
    LonM=np.unique(stloHD)
    VphM=np.empty((len(LatM),len(LonM)));VphM[:]=np.nan
    
    for k in tqdm(range(len(vgrHD_ma.reshape(-1)))):    
        iLon=np.where(LonM==stloHD[k])
        iLat=np.where(LatM==stlaHD[k])
        
        VphM[iLat,iLon]=vgrHD_ma.reshape(-1)[k] 
        
        
    #%
    MapWUS(LonM,LatM,VphM,VMIN,VMAX,cmnew)
    # MapWUS(LonM,LatM,VphM,VMIN,VMAX,'Greys')

    plt.title('v%s,%.2fs'%(t_type,per))
    
    dir_savefig='/uufs/chpc.utah.edu/common/home/u1318104/Figures/Zanskar_Meadow/07222024_Vgr/'
    plt.savefig(dir_savefig+'V%s_%.1fs_cri_p25km.png'%(t_type,per),transparent=True)
    # plt.savefig(dir_savefig+'un_V%s_%.1fs_cri_p25km.png'%(t_type,per),transparent=True)
    # plt.savefig(dir_savefig+'V%s_%.1fs_cri_p25km.png'%(t_type,per),transparent=True)


# plt.savefig('/uufs/chpc.utah.edu/common/home/u1318104/Figures/03272024_MeadowPre/%.2fs_vgr.png'%per,transparent=True)

