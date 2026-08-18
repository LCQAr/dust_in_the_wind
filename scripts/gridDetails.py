#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 11:03:08 2024

@author: leohoinaski
"""

import numpy as np
import pyproj
from shapely.geometry import Polygon
import netCDF4 as nc
import geopandas as gpd
import netCDFcreator as ncCreate
from datetime import timedelta
import os
import pandas as pd
import wrf
import ismember

def finder(a, b):
    # Step 1: Compute the absolute differences
    diff = np.abs(a[:, np.newaxis] - b)
    
    # Step 2: Find the index of the minimum difference along the axis
    closest_index = np.argmin(diff, axis=1)

    # If you want the actual closest values from `b`, you can use:
    closest_values = b[closest_index]

    return closest_values,closest_index

def createDomainShp(mcipGRIDDOT2Dpath,wrfoutPath):
    """
    Esta função é utilizada para gerar o geodataframe com o domínio de modelagem
    e extrair as coordenadas do arquivo do WRF.

    Parameters
    ----------
    wrfoutPath : path
        Caminho para o arquivo do WRF.    # abre o arquivo METCROD3D
    dsGRIDDOT2D = nc.Dataset(mcipGRIDDOT2Dpath)
    #latMCIP = dsGRIDDOT2D['LATD'][0,0,:,:]
    #lonMCIP = dsGRIDDOT2D['LOND'][0,0,:,:]
    xv,yv,lonGRIDDOT,latGRIDDOT = ioapiCoords(dsGRIDDOT2D)
    lonMCIP,latMCIP = eqmerc2latlon(dsGRIDDOT2D,xv,yv)

    Returns
    -------
    domainShp : geoDataFrame
        Geodataframe com a geometria do domínio de modelagem.
    lat : numpy array
        Matriz de latitudes.
    lon : numoy array
        matriz de longitudes.

    """
    # abre o arquivo METCROD3D
    dsGRIDDOT2D = nc.Dataset(mcipGRIDDOT2Dpath)
    #latMCIP = dsGRIDDOT2D['LATD'][0,0,:,:]
    #lonMCIP = dsGRIDDOT2D['LOND'][0,0,:,:]
    xv,yv,lonGRIDDOT,latGRIDDOT = ioapiCoords(dsGRIDDOT2D)
    lonMCIP,latMCIP = eqmerc2latlon(dsGRIDDOT2D,xv,yv)
    
    
    # Abringo arquivo do WRF
    ds = nc.Dataset(wrfoutPath)
    #Extraindo latitudes e longitudes em graus
    latWRF = ds['XLAT'][0,:,:]
    lonWRF = ds['XLONG'][0,:,:]

    lat,lat_index = finder(latMCIP[:,0], latWRF[:,0])
    lon,lon_index = finder(lonMCIP[0,:], lonWRF[0,:])
    
    # Criando o retângulo do domínio
    lat_point_list = [lat.min(), lat.max(), lat.max(), lat.min(), lat.min()]
    lon_point_list = [lon.min(), lon.min(), lon.max(), lon.max(), lon.min()]
    
    # Criando a geometria shapely 
    polygon_geom = Polygon(zip(lon_point_list, lat_point_list))
    
    # Colocando a geometria em um geodataframe
    domainShp = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[polygon_geom])   
    
    lon,lat = np.meshgrid(lon,lat)
    
    return  domainShp,lat,lon,lat_index,lon_index

def ioapiCoords(ds):
    # Latlon
    lonI = ds.XORIG
    latI = ds.YORIG
    print('lonI = '+str(lonI))
    print('latI = '+str(latI))
    
    # Cell spacing 
    xcell = ds.XCELL
    ycell = ds.YCELL
    ncols = ds.NCOLS
    nrows = ds.NROWS
    
    lon = np.arange(lonI,(lonI+ncols*xcell),xcell)
    lat = np.arange(latI,(latI+nrows*ycell),ycell)
    
    xv, yv = np.meshgrid(lon,lat)
    return xv,yv,lon,lat

def eqmerc2latlon(ds,xv,yv):

    mapstr = '+proj=merc +a=%s +b=%s +lat_ts=0 +lon_0=%s' % (
              6370000, 6370000, ds.XCENT)
    #p = pyproj.Proj("+proj=merc +lon_0="+str(ds.P_GAM)+" +k=1 +x_0=0 +y_0=0 +a=6370000 +b=6370000 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs")
    p = pyproj.Proj(mapstr)
    xlon, ylat = p(xv, yv, inverse=True)
    return xlon,ylat

def eqmerc2latlonMETCROD(ds,xv,yv):

    mapstr = '+proj=merc +a=%s +b=%s +lat_ts=0 +lon_0=%s' % (
              6370000, 6370000, ds.XCENT)
    #p = pyproj.Proj("+proj=merc +lon_0="+str(ds.P_GAM)+" +k=1 +x_0=0 +y_0=0 +a=6370000 +b=6370000 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs")
    p = pyproj.Proj(mapstr)
    xlon, ylat = p(xv-ds.XCELL/2, yv-ds.YCELL/2, inverse=True)
    return xlon,ylat


def gridding(lat,lon):
    # Extraindo os cantos das longitudes e latitudes
   
    # Inicializando a grid
    grids=[]
    
    lat=lat[:,0]
    lon=lon[0,:]
    # Loop para cada longitude
    for ii in range(1,lon.shape[0]):
        
        #Loop over each cel in y direction
        for jj in range(1,lat.shape[0]):
            
            #Criando retângulo de de cada célula
            lat_point_list = [lat[jj-1], lat[jj], lat[jj], lat[jj-1]]
            lon_point_list = [lon[ii-1], lon[ii-1], lon[ii], lon[ii]]
            
            # Criando um polígono para cada celula
            cel = Polygon(zip(lon_point_list, lat_point_list))
            grids.append(cel)
    
    
    return grids

def main(mcipMETCRO3Dpath,mcipGRIDDOT2Dpath,wrfoutFolder,domain):
    
    # abre o arquivo METCROD3D
    dsMETCRO3D = nc.Dataset(mcipMETCRO3Dpath)
    xv,yv,lonMETCROD,latMETCROD = ioapiCoords(dsMETCRO3D)
    lonMCIPMETCRO,latMCIPMETCROD = eqmerc2latlonMETCROD(dsMETCRO3D,xv,yv)
    
    # Extrai as datas do arquivo do MCIP
    datesTimeMCIP = ncCreate.datePrepCMAQ(dsMETCRO3D)
    print(datesTimeMCIP.shape)
    
    # Deinindo os arquivos do WRF que serão abertos. Devem ser compatíveis com 
    # as datas do respectivo arquivo do MCIP. 
    # file = [i for i in os.listdir(wrfoutFolder) if os.path.isfile(os.path.join(wrfoutFolder,i)) and \
    #          'wrfout_'+domain+'_'+str(datesTimeMCIP.year[0]).zfill(4)+'-'+\
    #              str(datesTimeMCIP.month[0]).zfill(2)+'-'+\
    #                  str(datesTimeMCIP.day[0]).zfill(2) in i]
    print('wrfout_'+domain+'_'+str((datesTimeMCIP.datetime[0]- timedelta(days=1)).year).zfill(4)+'-'+\
                 str((datesTimeMCIP.datetime[0]- timedelta(days=1)).month).zfill(2)+'-'+\
                     str((datesTimeMCIP.datetime[0]- timedelta(days=1)).day).zfill(2))
    # file = [i for i in os.listdir(wrfoutFolder) if os.path.isfile(os.path.join(wrfoutFolder,i)) and \
    #          'wrfout_'+domain+'_'+str((datesTimeMCIP.datetime[0]- timedelta(days=1)).year).zfill(4)+'-'+\
    #              str((datesTimeMCIP.datetime[0]- timedelta(days=1)).month).zfill(2)+'-'+\
    #                  str((datesTimeMCIP.datetime[0]- timedelta(days=1)).day).zfill(2) in i]
    files=['wrfout_'+domain+'_'+str((datesTimeMCIP.datetime[0]).year).zfill(4)+'-'+\
                     str((datesTimeMCIP.datetime[0]).month).zfill(2)+'-'+\
                         str((datesTimeMCIP.datetime[0]).day).zfill(2)+'_00:00:00',\
               'wrfout_'+domain+'_'+str((datesTimeMCIP.datetime[0]+ timedelta(days=1)).year).zfill(4)+'-'+\
                     str((datesTimeMCIP.datetime[0]+ timedelta(days=1)).month).zfill(2)+'-'+\
                         str((datesTimeMCIP.datetime[0]+ timedelta(days=1)).day).zfill(2)+'_00:00:00']
    
    
    # move para a pasta do WRF
    os.chdir(wrfoutFolder)
    #wrfoutPath = wrfoutFolder+'/'+file[0]
    wrfoutPath = wrfoutFolder+'/'+files[0]
    
    # abre os arquivos do WRFgrd
    ds = nc.MFDataset(files)
    
    # Extraindo latitudes e longitudes em graus
    # lat = ds['XLAT'][0,:-1,:-1]
    # lon = ds['XLONG'][0,:-1,:-1]
    
    # extrai as datas dos arquivos do WRF que foram abertos
    datesTime = ncCreate.datePrepWRF(pd.to_datetime(wrf.extract_times(ds,
                                                                      wrf.ALL_TIMES)))
    #datesTime = pd.to_datetime(ds['Time'].values)
    
    # identifica datas coincidentes no MCIP e WRF
    #print(datesTime.shape)
    lia, loc = ismember.ismember(np.array(datesTime.datetime), 
                                 np.array(datesTimeMCIP.datetime))
    
    # lialat, loclat = ismember.ismember(lat[:,0],ylat[:,0])
    # lialon, loclon = ismember.ismember(lon[0,:],xlon[0,:])
    
    domainShp,lat,lon,lat_index,lon_index = createDomainShp(mcipGRIDDOT2Dpath,wrfoutPath)
    grids = gridding(lat,lon)
    
    return ds,datesTime,lia,domainShp,lat,lon,lat_index,lon_index,grids