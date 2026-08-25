#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 13:51:13 2025

@author: lcqar
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import temporalStatistics as tst
from shapely import Point


#%%

lista_ds = []

for mes in range(12):
    if mes in [0,2,4,6,7,9]:
        dias = 31
    elif mes in [3,5,8,10,11]:
        dias = 30
    else:
        dias = 28
    
    lista_mes = []
        
    for dia in range(dias):
        
        dia_inicial=str(mes+1).zfill(2)+'-'+str(dia+1).zfill(2)
               
        if dia == dias-1:
            if mes == 11:
                dia_final=str(mes+1).zfill(2)+'-'+str(dia+2).zfill(2)
            else:
                dia_final=str(mes+2).zfill(2)+'-'+str(1).zfill(2)
        else:
            dia_final=str(mes+1).zfill(2)+'-'+str(dia+2).zfill(2)

        print(dia_inicial+'  '+dia_final)

        ds_pm10 = xr.open_dataset('/home/lcqar/MMA/windBlowDustBR/Outputs/windBlowDust_PM10_2023-'+dia_inicial+'-00:00:00_2023-'+dia_final+'-00:00:00.nc')
        
        var = ds_pm10['PM10']
        
        lista_mes.append(var)
        
    lista_ds.append(lista_mes)

#%%

def latlon_2d(data):
   
    """
    Function to convert latlon from NetCDF into 2D arrays.

    Parameters:
    ----------
    dir_data : str
        Path to the NetCDF file containing the data with the coordinates.

    Returns:
    -------
    lon2d : numpy.ndarray
        1D array containing the extracted and transformed longitude coordinates.
    
    lat2d : numpy.ndarray
        1D array containing the extracted and transformed latitude coordinates.

    Dependencies:
    -------------
    - xarray (xr)
    - tst (must contain the functions ioapiCoords and eqmerc2latlon)
    """

    #read data
    xv, yv, lon, lat = tst.ioapiCoords(data)
    xlon, ylat = tst.eqmerc2latlon(data, xv, yv)
    lon2d = xlon.flatten()
    lat2d = ylat.flatten()
    
    return lon2d, lat2d

#%%

lista_gdf = []

lon, lat = latlon_2d(ds_pm10)
s = gpd.GeoSeries(map(Point, zip(lon.flatten(), lat.flatten())))

for mes in range(12):
    if mes in [0,2,4,6,7,9]:
        dias = 31
    elif mes in [3,5,8,10,11]:
        dias = 30
    else:
        dias = 28
    
    lista_mes = []
        
    for dia in range(dias):
        
        dia_inicial=str(mes+1).zfill(2)+'-'+str(dia+1).zfill(2)
               
        if dia == dias-1:
            if mes == 11:
                dia_final=str(mes+1).zfill(2)+'-'+str(dia+2).zfill(2)
            else:
                dia_final=str(mes+2).zfill(2)+'-'+str(1).zfill(2)
        else:
            dia_final=str(mes+1).zfill(2)+'-'+str(dia+2).zfill(2)

        print(dia_inicial+'  '+dia_final)

        ds_pm10 = xr.open_dataset('/home/lcqar/MMA/windBlowDustBR/Outputs/updated/windBlowDust_PM10_2023-'+dia_inicial+'-00:00:00_2023-'+dia_final+'-00:00:00.nc')
        
        var = ds_pm10['PM10']
        
        lista_dia = []
        
        for h in range(24):
        
            emis = var[h,:,:].values.flatten()
                
            lista_dia.append(gpd.GeoDataFrame(
                {'emissao' : emis},
                geometry=s.geometry,
                crs = 'EPSG:4674'))
        
        lista_mes.append(lista_dia)
        
    lista_gdf.append(lista_mes)

shp = gpd.read_file('/home/lcqar/Congonhas/E04/shapefiles/congonhas.shp')

#%% Plotando no espaço

'''
    Figura com 12 imagens para soma de cada mês
    Preparar com figura de Congonhas
'''

def fig_mes_gdf(lista_gdf):
    
    somas_mensais = []

    for mes in range(12):
        gdf_mes = None  # acumulador
    
        for dia in range(len(lista_gdf[mes])):  # normalmente 30
            for hora in range(24):
                gdf_hora = lista_gdf[mes][dia][hora]
    
                if gdf_mes is None:
                    gdf_mes = gdf_hora.copy()
                else:
                    gdf_mes['emissao'] += gdf_hora['emissao']
    
        somas_mensais.append(gdf_mes)
            
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    # Obter valores mínimos e máximos para normalizar escala
    valores = [gdf['emissao'].values for gdf in somas_mensais]
    vmin = 0
    vmax = max(gdf['emissao'].max() for gdf in somas_mensais)
    
    for mes in range(12):
        ax = axes[mes]
        gdf = somas_mensais[mes]
    
        gdf.plot(
            column='emissao',
            ax=ax,
            cmap='jet',
            norm=LogNorm(vmin=vmin+1e-6, vmax=vmax),
            markersize=1,
            legend=True
        )
    
        ax.set_title(f'Mês {mes+1}')
        ax.set_axis_off()
    
    plt.tight_layout()
    plt.show()
    
#%%

def fig_mes_ds(lista_ds):
    
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))  # 4 linhas, 3 colunas
    axes = axes.flatten()  # Facilita acessar os subplots como uma lista
    
    for mes in range(12):
        
        # Empilhar os DataArrays do mês
        ds = xr.concat(lista_ds[mes], dim="EMPILHAR")
        
        # Nome da primeira dimensão (geralmente 25)
        primeira_dim = ds.dims[1]
        
        # Soma ao longo de EMPILHAR e da primeira dimensão
        ds_soma = ds.sum(dim=["EMPILHAR", primeira_dim])
        
        # Plotar no subplot correspondente
        ax = axes[mes]
        im = ds_soma.plot(ax=ax,
                           cmap="jet",
                           norm=LogNorm(vmin=ds_soma.min().values + 1, vmax=ds_soma.max().values),
                           add_colorbar=False)
        
        ax.set_title(f"Mês {mes + 1}")
        ax.set_xlabel("")
        ax.set_ylabel("")
    
    plt.tight_layout()
    plt.show()
    
#%% Plotando soma em torno de Congonhas no tempo

'''
    Figura com shade e linha média
'''

def plot_no_ano(lista_gdf, shp):
    
    somas=[]

    buffer = shp.buffer(0.35)

    dominio = lista_gdf[0][0][0]
    interseccao = dominio.intersects(buffer.geometry[0])

    for mes in range(12):
        for dia in range(len(lista_gdf[mes])):
            soma=0
            for hora in range(24):
                dominio = lista_gdf[mes][dia][hora]
                soma += dominio[interseccao].emissao.sum()
            somas.append(60*60*soma/1000)
    
    x = pd.date_range(start='2023-01-01 00:00:00', periods=len(somas), freq='D')

    # Plot
    plt.figure(figsize=(10, 6))
    
    # Dados
    plt.plot(x, somas, color='blue', linewidth=1)
    
    # Rótulos e título
    plt.xlabel("Data (horas)")
    plt.ylabel("Emissão (kg)")
    plt.yscale("log")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(True)
    
    plt.show()

#%% Plotando por hora

'''
    Figura com linha média e percentis nas horas do dia
'''

#%%


