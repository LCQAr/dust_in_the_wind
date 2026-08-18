#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
----------------------------mainWindBlowDust.py-------------------------------

Este script em python é utilizado para rodar o módulo windBlowDust e gerar os 
inputs para o CMAQ. 
   

Created on Fri Mar 15 16:04:37 2024

Referências:
    https://agupubs.onlinelibrary.wiley.com/doi/10.1002/2016MS000823
    https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2010JD014649

@author: leohoinaski

"""

import regridMAPBIOMAS as regMap
import soilPrep as sp
import metPrep as mp
import windBlowDustCalc as wbd
import netCDFcreator as ncCreate
import os
import numpy as np
#import geopandas as gpd
import netCDF4 as nc
import wrf
import pandas as pd
from datetime import timedelta
import ismember
import windBlowDustSpeciation as wbds
import gridDetails as grd


# Dicionários de poluentes
PM25 = {
  "Unit": '$\g.S^{-1}$',
  "tag":'PMFINE',
  "range":[0,2.5] # micrometers
}

PMC = {
  "Unit": '$\g.S^{-1}$',
  "tag":'PMC',
  "range":[2.5,10] # micrometers
}

PM10 = {
  "Unit": '$\g.S^{-1}$',
  "tag":'PM10',
  "range":[0,10] # micrometers
}

PM1 = {
  "Unit": '$\g.S^{-1}$',
  "tag":'PMULTRAFINE',
  "range":[0,1] # micrometers
}

ALL = {
  "Unit": '$\g.S^{-1}$',
  "tag":'AllFractions',
  "fractions":['PMFINE','PMC','PM10'] # micrometers
}

# Inputs
domain = 'd02'
GDNAM = 'MG_3km'
RESET_GRID = False
year = 2021


# Definindo o caminho para as pastas
rootFolder =  os.path.dirname(os.path.dirname(os.getcwd()))

#wrfoutFolder = rootFolder+'/BR_2019'
#wrfoutFolder='/home/lcqar/CMAQ_REPO/data/WRFout/BR/WRFd01_BR_20x20'
#wrfoutFolder='/home/WRFout/share/Congonhas/2021/d02'
#mcipPath='/home/artaxo/CMAQ_REPO/data/mcip/'+GDNAM
#wrfoutFolder='/media/leohoinaski/HDD/MG_3km'
#mcipPath='/media/leohoinaski/HDD/MG_3km'
wrfoutFolder='/mnt/sdb1/MG_3km'
mcipPath='/mnt/sdb1/MG_3km'


mcipMETCRO3Dpath = mcipPath+'/METCRO3D_'+GDNAM+'.nc'
mcipGRIDDOT2Dpath = mcipPath+'/GRIDDOT2D_'+GDNAM+'.nc'
windBlowDustFolder = os.path.dirname(os.getcwd())
#wrfoutFolder='/home/lcqar/CMAQ_REPO/data/WRFout/BR/WRFd01_BR_20x20'
#mcipMETCRO3Dpath ='/home/lcqar/CMAQ_REPO/data/mcip/BR_2019/METCRO3D_BR_2019.nc'

inputFolder = os.path.dirname(os.getcwd())+'/inputs'
tablePath = os.path.dirname(os.getcwd())+'/inputs/tables'
outfolder = os.path.dirname(os.getcwd())+'/Outputs/'+GDNAM

# Definição dos ids do MAPBIOMAS que serão utilizados na estimativa 
# das emissões no windblowdust
idSoils = [23,30,25] #4.1. Praia, Duna e Areal  4.3. Mineração 4.4. Outras Áreas não Vegetadas

# espaçamento entre diâmetros para integração dos valores
dx = 0.1

# frações que serão calculadas
Fractions = [PM25,PMC] # Lista com tipo de emissão por diâmetro. 
                        #Não precisa incluir o PM10 se já tiver PM25 e PM10

# condição para verificar se a pasta de output existe
if os.path.isdir(outfolder):
    print('You have the outputs folder')
else:
    os.makedirs(outfolder, exist_ok=True)


# Grid setup
ds,datesTime,lia,domainShp,lat,lon,lat_index,lon_index,grids = grd.main(
    mcipMETCRO3Dpath,mcipGRIDDOT2Dpath,wrfoutFolder,domain)


# Regrid MAPBIOMAS
av,al,alarea,lat,lon,domainShp = regMap.main(GDNAM,inputFolder,
                                             outfolder,year,idSoils,RESET_GRID,
                                             grids,domainShp,lat,lon)

# loop para cada fração do PM
for EmisD  in Fractions:
    
    # determina os ranges das particulas - min e max
    Dmax = np.max(EmisD['range'])
    Dmin = np.min(EmisD['range'])
    
    # cria um arranjo de diâmetros de particulas para integração
    diam = np.arange(np.min(EmisD['range']),np.max(EmisD['range'])+dx,dx)
    
    # inicializa a variável FdustTotal que acumulará as estimativas para 
    # cada diâmetro
    FdustTotal=[]
    
    # seleciona os valores dentro da faixa da fração ex-0 a 2.5 micrometros
    diamSelect = diam[(Dmin<=diam) & (Dmax>=diam)]
    
    # loop em cada diâmetro
    for jj,diameters in enumerate(diamSelect):
        
        print(diameters)
        
        # executa a função soilPrep
        clayRegrid,sRef = sp.main(inputFolder,outfolder,domainShp,GDNAM,
                                  lat,lon,diameters,RESET_GRID,grids)
        
        # executa a função metPrep
        ustar,ustarT,ustarTd,avWRF,ustarWRF = mp.main(ds,tablePath,av,al,
                                                      diameters,clayRegrid,lia,
                                                      lat_index,lon_index)
        
        # executa a função windBlowDustCalc
        Fdust,Fhd,Fhtot,Fvtot = wbd.wbdFlux(avWRF,alarea,sRef,clayRegrid,
                                            ustar,ustarT,ustarTd)
        
        # já rodou uma vez, logo, não precisa resetar os arquivos 
        # intermediários
        RESET_GRID = False
        
        # acumula os valores em cada diâmetro
        FdustTotal.append(Fdust)
    
    # empilha os valores em um array numpy
    FdustTotal = np.stack(FdustTotal)
    
    # estima a massa total de particulas dentro da faixa da fração
    # faz a integral dos dados estimados
    #FdustD = np.trapz(FdustTotal,dx=dx, axis=0)   
    
    # faz a média do fluxo para cada diâmetro
    FdustD = np.nanmedian(FdustTotal, axis=0)   
    print(FdustD.shape)
    print(np.nanmax(FdustD))
    
    # cria o netCDF com a estimativa das particulas
    ncCreate.createNETCDFtemporal(outfolder,'windBlowDust_',FdustD,
                                  datesTime[lia],mcipMETCRO3Dpath,EmisD)
    
    # faz a especiação química das particulas
    if EmisD==PM25:
        FdustFINE = FdustD
        print('FdustFINE max: '+str(FdustFINE.max()))
        FdustFINESpec = wbds.speciate(windBlowDustFolder, FdustFINE)
    elif EmisD==PMC:
        FdustCOARSE = FdustD
        FdustCOARSEpec = wbds.speciate(windBlowDustFolder, FdustCOARSE)
        print('FdustCOARSE max: '+str(FdustCOARSE.max()))
    else:
        print('You have selected an awkward fraction')
    
    # se existir as parcelas PMFINE e PMC, calcula o PM10
    try:
        FdustPM10 = FdustFINE+FdustCOARSE
        ncCreate.createNETCDFtemporal(outfolder,'windBlowDust_',FdustPM10,
                                      datesTime[lia],mcipMETCRO3Dpath,PM10)
    except:
        print('You do not have the fractions required for PM10')   
        
# Acumula todas as estimativas de particulas sem especiação        
FdustALL = [FdustFINE,FdustCOARSE,FdustPM10]
FdustALL = np.stack(FdustALL,axis=0)
print('FdustALL max: '+str(FdustALL.max()))

# cria o netCDF com todas as especies de particulas/frações
ncCreate.createNETCDFtemporal(outfolder,'windBlowDust_',FdustALL,
                              datesTime[lia],mcipMETCRO3Dpath,ALL)

# soma as emissões de cada especie no PM25 e PMC 
#FdustSpeciated = FdustFINESpec + FdustCOARSEpec
FdustSpeciated = FdustFINESpec

print('FdustSpeciated max: '+str(FdustSpeciated.max()))

# cria o netCDF especiado
ncCreate.createNETCDFtemporalSpeciated(windBlowDustFolder,outfolder,
                                       'windBlowDust_',FdustSpeciated,
                                       datesTime[lia],mcipMETCRO3Dpath)

#%%
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import geopandas as gpd
shape_path= rootFolder+'/shapefiles/BR_regions.shp'   
borderShape = gpd.read_file(shape_path)

# ustar no espaço
fig, ax = plt.subplots()
pcm= ax.pcolor(lon[:-1,:-1],lat[:-1,:-1], np.nanmean(ustar[:, :, :],axis=0))
borderShape.boundary.plot(edgecolor='black',linewidth=0.5,ax=ax)
ax.set_title('ustar')
ax.set_xlim([lon.min(),lon.max()])
ax.set_ylim([lat.min(),lat.max()])
ax.set_frame_on(False)
ax.set_xticks([])
ax.set_yticks([])
cbar = fig.colorbar(pcm, ax=ax,fraction=0.04, pad=0.02,
                        #extend='both', 
                        #ticks=bounds,
                        #spacing='uniform',
                        orientation='horizontal',)



# ustarWRF no espaço
fig, ax = plt.subplots()
ax.pcolor(lon[:-1,:-1],lat[:-1,:-1], np.nanmean(ustarWRF[:, :, :].data,axis=0))
pcm = borderShape.boundary.plot(edgecolor='black',linewidth=0.5,ax=ax)
ax.set_title('ustarWRF')
ax.set_xlim([lon.min(),lon.max()])
ax.set_ylim([lat.min(),lat.max()])
ax.set_frame_on(False)
ax.set_xticks([])
ax.set_yticks([])
# cbar = fig.colorbar(pcm,ax=ax,fraction=0.04, pad=0.02,
#                         #extend='both', 
#                         #ticks=bounds,
#                         #spacing='uniform',
#                         orientation='horizontal',)


# avWRF no espaço
fig, ax = plt.subplots()
pcm = ax.pcolor(lon[:-1,:-1],lat[:-1,:-1],np.nansum(avWRF,axis=0))
borderShape.boundary.plot(edgecolor='black',linewidth=0.5,ax=ax)
ax.set_title('avWRF')
ax.set_xlim([lon.min(),lon.max()])
ax.set_ylim([lat.min(),lat.max()])
ax.set_frame_on(False)
ax.set_xticks([])
ax.set_yticks([])
cbar = fig.colorbar(pcm, ax=ax,fraction=0.04, pad=0.02,
                        #extend='both', 
                        #ticks=bounds,
                        #spacing='uniform',
                        orientation='horizontal',)


fig, ax = plt.subplots()
pcm = ax.pcolor(lon[:-1,:-1],lat[:-1,:-1],sRef[:, :])
borderShape.boundary.plot(edgecolor='black',linewidth=0.5,ax=ax)
ax.set_title('sRef')
ax.set_xlim([lon.min(),lon.max()])
ax.set_ylim([lat.min(),lat.max()])
ax.set_frame_on(False)
ax.set_xticks([])
ax.set_yticks([])
cbar = fig.colorbar(pcm, ax=ax,fraction=0.04, pad=0.02,
                        #extend='both', 
                        #ticks=bounds,
                        #spacing='uniform',
                        orientation='horizontal')

fig, ax = plt.subplots()
pcm = ax.pcolor(lon[:-1,:-1],lat[:-1,:-1],np.nansum(alarea[:,:, :],axis=0))
borderShape.boundary.plot(edgecolor='black',linewidth=0.5,ax=ax)
ax.set_title('alarea')
ax.set_xlim([lon.min(),lon.max()])
ax.set_ylim([lat.min(),lat.max()])
#ax.set_frame_on(False)
ax.set_xticks([])
ax.set_yticks([])
cbar = fig.colorbar(pcm, ax=ax,fraction=0.04, pad=0.02,
                        #extend='both', 
                        #ticks=bounds,
                        #spacing='uniform',
                        orientation='horizontal',)


fig, ax = plt.subplots()
pcm = ax.pcolor(lon[:-1,:-1],lat[:-1,:-1],clayRegrid[0,:,:])
borderShape.boundary.plot(edgecolor='black',linewidth=0.5,ax=ax)
ax.set_title('clayRegrid')
ax.set_xlim([lon.min(),lon.max()])
ax.set_ylim([lat.min(),lat.max()])
#ax.set_frame_on(False)
ax.set_xticks([])
ax.set_yticks([])
cbar = fig.colorbar(pcm, ax=ax,fraction=0.04, pad=0.02,
                        #extend='both', 
                        #ticks=bounds,
                        #spacing='uniform',
                        orientation='horizontal',)



fig, ax = plt.subplots()
pcm = ax.pcolor(lon,lat,np.nanmean(Fvtot,axis=0))
borderShape.boundary.plot(edgecolor='black',linewidth=0.5,ax=ax)
ax.set_title('Fvtot')
ax.set_xlim([lon.min(),lon.max()])
ax.set_ylim([lat.min(),lat.max()])
#ax.set_frame_on(False)
ax.set_xticks([])
ax.set_yticks([])
cbar = fig.colorbar(pcm, ax=ax,fraction=0.04, pad=0.02,
                        #extend='both', 
                        #ticks=bounds,
                        #spacing='uniform',
                        orientation='horizontal',)


fig, ax = plt.subplots()
pcm = ax.pcolor(lon,lat,np.nanmean(Fhtot,axis=0))
borderShape.boundary.plot(edgecolor='black',linewidth=0.5,ax=ax)
ax.set_title('Fhtot')
ax.set_xlim([lon.min(),lon.max()])
ax.set_ylim([lat.min(),lat.max()])
#ax.set_frame_on(False)
ax.set_xticks([])
ax.set_yticks([])
cbar = fig.colorbar(pcm, ax=ax,fraction=0.04, pad=0.02,
                        #extend='both', 
                        #ticks=bounds,
                        #spacing='uniform',
                        orientation='horizontal',)


fig, ax = plt.subplots()
pcm = ax.pcolor(lon,lat,np.nansum(FdustD[:, :, :], axis=0),norm=colors.LogNorm())
borderShape.boundary.plot(edgecolor='black',linewidth=0.5,ax=ax)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim([lon.min(),lon.max()])
ax.set_ylim([lat.min(),lat.max()])
cbar = fig.colorbar(pcm, ax=ax,fraction=0.04, pad=0.02,
                        #extend='both', 
                        #ticks=bounds,
                        #spacing='uniform',
                        orientation='horizontal',)
cbar.ax.set_xlabel(' FdustD Wind blow Dust emission\n (g/s)', rotation=0,fontsize=8)
ax.set_frame_on(False)
cbar.ax.tick_params(labelsize=6) 
fig.tight_layout()
#ax.set_title('FdustD')

fig, ax = plt.subplots()
ax.scatter(ustar.flatten(),FdustD.flatten())
ax.set_title('FdustD vs ustar')
ax.set_yscale('log')
#https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2010JD014649


fig, ax = plt.subplots()
ax.scatter(ustar.flatten(),Fvtot.flatten())
ax.set_title('Fvtot vs ustar')
ax.set_yscale('log')
#https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2010JD014649

fig, ax = plt.subplots()
ax.scatter(ustar.flatten(),Fhtot.flatten())
ax.set_title('Fhtot vs ustar')
ax.set_yscale('log')


fig, ax = plt.subplots()
ax.scatter(alarea[1,:,:].repeat(Fvtot.shape[0]).flatten(),Fvtot.flatten())
ax.set_title('Fvtot vs alarea')
ax.set_yscale('log')


fig, ax = plt.subplots()
pcm = ax.pcolor(lon,lat,np.nansum(FdustFINE[:, :, :], axis=0),norm=colors.LogNorm())
borderShape.boundary.plot(edgecolor='black',linewidth=0.5,ax=ax)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim([lon.min(),lon.max()])
ax.set_ylim([lat.min(),lat.max()])
# cbar = fig.colorbar(pcm, ax=ax,fraction=0.04, pad=0.02,
#                         #extend='both', 
#                         #ticks=bounds,
#                         #spacing='uniform',
#                         orientation='horizontal',)

cbar.ax.set_xlabel('PMFINE'+'\nWind blow Dust emission\n (g/s)', rotation=0,fontsize=8)
ax.set_frame_on(False)
cbar.ax.tick_params(labelsize=6) 
fig.tight_layout()
#ax.set_title('FdustD')

fig, ax = plt.subplots()
pcm = ax.pcolor(lon,lat,np.nanmax(FdustCOARSE[:, :, :], axis=0),norm=colors.LogNorm())
borderShape.boundary.plot(edgecolor='black',linewidth=0.5,ax=ax)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim([lon.min(),lon.max()])
ax.set_ylim([lat.min(),lat.max()])
cbar = fig.colorbar(pcm, ax=ax,fraction=0.04, pad=0.02,
                        #extend='both', 
                        #ticks=bounds,
                        #spacing='uniform',
                        orientation='horizontal',)

cbar.ax.set_xlabel('FdustCOARSE'+'\nWind blow Dust emission\n (g/s)', rotation=0,fontsize=8)
ax.set_frame_on(False)
cbar.ax.tick_params(labelsize=6) 
fig.tight_layout()
#ax.set_title('FdustD')
