#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Created on Tue Mar 19 12:58:58 2024

This class is used to prepare the soil properties for windBlowDuscCalc

Inputs : http://geoinfo.cnps.embrapa.br/documents/3295
        https://geoftp.ibge.gov.br/informacoes_ambientais/pedologia/vetores/brasil_5000_mil/
        https://geo.anm.gov.br/portal/apps/webappviewer/index.html?id=6a8f5ccc4b6a4c2bba79759aa952d908
        
        Vasques, G.M., Coelho, M.R., Dart, R.O., Cintra, L.C., Baca, J.F.M. 
        (2021). Soil Clay, Silt and Sand Content Maps for Brazil at 0-5, 5-15,
        15-30, 30-60, 60-100 and 100-200 cm Depth Intervals with 90 m Spatial 
        Resolution. Version 2021. Embrapa Solos, Rio de Janeiro, Brazil.
        
@author: leohoinaski


"""

import pandas as pd
import numpy as np
import os
import netCDF4 as nc
import rioxarray as riox
import regridMAPBIOMAS as regMap
from scipy import optimize,stats
from rasterio.enums import Resampling
from shapely.geometry import Polygon
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
from scipy.stats import lognorm
from scipy.integrate import simpson, trapezoid
#from dask.distributed import Client

def rasterLatLon(raster):
    """

    Parameters
    ----------

    raster : xarray
        rioxarray variable

    Returns
    -------
    x : longitude
        values depend on coordinate system
    y : latitude
        values depend on coordinate system.

    """
    
    # Reprojetando para o EPSG 4326
    raster = raster.rio.reproject("EPSG:4326")
    
    # Extraindo matriz de x
    x = raster.x.values
    
    # Extraindo matriz de y
    y = raster.y.values
    
    return x, y



def cutSoil(domainShp,inputFolder,outfolder,GRDNAM):
    """
    
    Esta função é utilizada para fazer reduzir a resolução do dado de clay 
    content. 
    
    Parameters
    ----------
    domainShp : TYPE
        DESCRIPTION.
    inputFolder : TYPE
        DESCRIPTION.
    outfolder : TYPE
        DESCRIPTION.
    GRDNAM : TYPE
        DESCRIPTION.

    Returns
    -------
    raster : TYPE
        DESCRIPTION.

    """
    
    # print('Starting cutSoil function - windBlowDust')
    
    lista_solos = []
    
    for soil in ['clay','silt','sand','bulk_density']:
    
        print('Abrindo raster de 0 a 5 cm')
        
        # Abrindo arquivo com o teor de argila
        #raster = riox.open_rasterio(inputFolder+'/br_clay_content_30-60cm_pred_g_kg/br_clay_content_30-60cm_pred_g_kg.tif', chunks={'x': 1024, 'y': 1024})
        raster_5 = riox.open_rasterio(inputFolder+'/br_clay_content_30-60cm_pred_g_kg/br_'+soil+'_content_0_5cm_pred_g_kg.tif', chunks={'x': 1024, 'y': 1024})
        
        print('abrindo raster de 5 a 15 cm')
        
        raster_15 = riox.open_rasterio(inputFolder+'/br_clay_content_30-60cm_pred_g_kg/br_'+soil+'_content_5_15cm_pred_g_kg.tif', chunks={'x': 1024, 'y': 1024})
        
        print('calculando raster de 0 a 15 cm')
        
        raster = (raster_5 + 2*raster_15)/3
        
        # Reduzindo a dimensão do raster 1/5
        downscale_factor = 1/5
        
        # nova largura e altura
        new_width = raster.rio.width * downscale_factor
        new_height = raster.rio.height * downscale_factor
        
        print('Reprojetando')
        
        # faz o downscaling
        raster = raster.rio.reproject(raster.rio.crs, shape=(int(new_height), 
                                                             int(new_width)),
                                      resampling=Resampling.bilinear)
        
        # VERIFICAR!!! conversão da unidade de g/kg para % 
        # https://angeo.copernicus.org/articles/17/149/1999/angeo-17-149-1999.pdf
        # equação 4 :https://agupubs.onlinelibrary.wiley.com/doi/10.1002/2016MS000823
        # dividido por 1000 para transformar g em kg e vezes 100 para colocar em %
        print('%')
        if soil != 'bulk_density':
            print(soil)
            raster = (raster/1000)*100  
        elif soil == 'bulk_density':
            print(soil)
            raster =  raster*1000
        else:
            print('Solo '+soil+' não existe')
            return
        
        print('Corrigindo falhas')
        # remove os valores faltantes iguais a -999
        raster = raster.where(raster>0)
        print('raster corrigido')
        lista_solos.append(raster)
            
    print('1')
    
    return lista_solos


def rasterInGrid(domainShp,raster,x,y,lat,lon,grids):
    """
    
    Esta função faz o regrid da matriz de clay content para o domínio de 
    modelagem.

    Parameters
    ----------
    domainShp : geodataframe
        geodataframe com o poligono do dominio de modelagem.
    raster : rioxarray
        matriz com os dados do raster clay content.
    x : np.array
        matriz de x do raster clay content.
    y : np.array
        matriz de y do raster clay content.
    lat : np.array
        matriz de latitudes do domínio.
    lon : np.array
        matriz de longitudes do domínio.
    grids : np.array
        geometrias das células.

    Returns
    -------
    matRegrid : np.array
        matriz com o regrid do clay content para o domínio de modelagem.

    """
    
    # Incializando a matriz de regrid do clay content
    matRegrid=np.empty((lat.shape[0]-1,lon.shape[1]-1))
    # Todos os valores como nan
    matRegrid[:,:] = np.nan
    # Definindo o EPSG
    raster = raster.rio.reproject("EPSG:4326")
    
    print('Clipando raster')
    
    clippedRaster = raster.rio.clip(domainShp.geometry)
    
    df = pd.DataFrame({'geometry':grids})
    gdf = gpd.GeoDataFrame(df, crs="EPSG:4326")
    gdf.set_geometry('geometry', inplace=True)
    
    # Recortando o tipo de solo para cada célula do domínio.
    clay=[]
    for index, row in gdf.iterrows():
        print(index)
        try:
            # @TODO: Checar alternativa
            clipped = clippedRaster.rio.clip(
                gpd.GeoDataFrame(index=[0], crs="EPSG:4326",geometry=[row.geometry]).geometry)
            clipped = np.array(clipped)
            if clipped.shape[0]>0:
               clay.append(np.nanmean(clipped))
            else:
               clay.append(np.nan)
        except:
            clay.append(np.nan)
    
    # fazendo o rashape
    matRegrid[:,:] = np.array(clay).reshape((lat.shape[1]-1,lon.shape[0]-1)).transpose() 

                
    # substitui nan por 0
    matRegrid[np.isnan(matRegrid)] = 0
    
    return matRegrid

def rasterInGrid_optimized(domainShp,raster,x,y,lat,lon,grids):
    """
    
    Esta função faz o regrid da matriz de clay content para o domínio de 
    modelagem.

    Parameters
    ----------
    domainShp : geodataframe
        geodataframe com o poligono do dominio de modelagem.
    raster : rioxarray
        matriz com os dados do raster clay content.
    x : np.array
        matriz de x do raster clay content.
    y : np.array
        matriz de y do raster clay content.
    lat : np.array
        matriz de latitudes do domínio.
    lon : np.array
        matriz de longitudes do domínio.
    grids : np.array
        geometrias das células.

    Returns
    -------
    matRegrid : np.array
        matriz com o regrid do clay content para o domínio de modelagem.

    """
    
    # Incializando a matriz de regrid do clay content
    matRegrid=np.empty((lat.shape[0]-1,lon.shape[1]-1))

    # Todos os valores como nan
    matRegrid[:,:] = np.nan

    # Definindo o EPSG
    raster = raster.rio.reproject("EPSG:4326")

    # Se tiver dimensão band unitária, remover
    if "band" in raster.dims and raster.sizes.get("band", 1) == 1:
        raster = raster.squeeze("band", drop=True)
    
    print('Clipando raster')
    
    clippedRaster = raster.rio.clip(domainShp.geometry)

    # Se tiver dimensão band unitária, remover
    if "band" in clippedRaster.dims and clippedRaster.sizes.get("band", 1) == 1:
        clippedRaster = clippedRaster.squeeze("band", drop=True)
    
    df = pd.DataFrame({'geometry':grids})
    gdf = gpd.GeoDataFrame(df, crs="EPSG:4326")
    gdf.set_geometry('geometry', inplace=True)

    # Criando índice
    gdf = gdf.reset_index(drop=True)
    gdf["zone_id"] = np.arange(1, len(gdf) + 1, dtype=np.int32)
    
    print('Rasterizando célula da grade')

    # Rasteriza todas as células de uma vez
    zone_raster = rasterize(
        shapes=zip(gdf.geometry, gdf["zone_id"]),
        out_shape=clippedRaster.shape,
        transform=clippedRaster.rio.transform(),
        fill=0,
        dtype="int32",
        all_touched=False
    )

    # Tratando nodata
    nodata = clippedRaster.rio.nodata
    if nodata is not None:
        clippedRaster = clippedRaster.where(clippedRaster != nodata, np.nan)

    # Pixels válidos: zona > 0 e valor não nan
    clipped_arr = clippedRaster.values
    valid = (zone_raster > 0) & (~np.isnan(clipped_arr))

    zone_ids = zone_raster[valid]
    values = clipped_arr[valid]

    print("Calculando médias por célula")

    # Soma e contagem por zone_id
    sums = np.bincount(zone_ids, weights=values)
    counts = np.bincount(zone_ids)

    clay = np.full(len(gdf), np.nan, dtype=float)
    valid_ids = np.where(counts > 0)[0]

    for zid in valid_ids:
        if zid == 0:
            continue
        clay[zid - 1] = sums[zid] / counts[zid]
    
    # fazendo o rashape
    matRegrid[:,:] = np.array(clay).reshape(
        (lat.shape[1]-1,lon.shape[0]-1)).transpose() 

    # substitui nan por 0
    matRegrid[np.isnan(matRegrid)] = 0
    
    return matRegrid

def find_nearest(array, value):
    """
    Encontra o indice da matriz que possui o valor mais próximo.

    Parameters
    ----------
    array : np.array
        array com matriz de valores.
    value : float
        valor para ser encontrado na matriz.

    Returns
    -------
    idx : int
        índice da matriz.

    """
    # transforma em np.array
    array = np.asarray(array)
    
    # Acha o indice
    idx = (np.abs(array - value)).argmin()
    
    return idx

def regridSoilTexture(outfolder,inputFolder,lat,lon,GDNAM,grids):
    """
    

    Parameters
    ----------
    outfolder : path
        caminho para a pasta de outputs.
    inputFolder : path
        caminho para a pasta de inputs.
    lat : np.array
        matriz de latitudes do domínio.
    lon : np.array
        matriz de longitudes do domínio.
    GDNAM : str
        nome do domínio conforme o MCIP.

    Returns
    -------
    matRegrid : np.array
        matriz com o regrid da textura do solo.

    """
    
  
    # # Extraindo os cantos das longitudes e latitudes
    # lonCorner = np.append(np.append(lon[0,:-1]- np.diff(lon[0,:])/2,lon[0,-1]),
    #                       lon[0,-1]+np.diff(lon[0,-3:-1])/2)
    
    # latCorner = np.append(np.append(lat[:-1,0]- np.diff(lat[:,0])/2,lat[-1,0]),
    #                       lat[-1,0]+np.diff(lat[-3:-1,0])/2)
    
    # # Inicializando a grid
    # grids=[]
    
    # # Loop para cada longitude
    # for ii in range(1,lonCorner.shape[0]):
        
    #     #Loop over each cel in y direction
    #     for jj in range(1,latCorner.shape[0]):
            
    #         #Criando retângulo de de cada célula
    #         lat_point_list = [latCorner[jj-1], latCorner[jj], latCorner[jj], latCorner[jj-1]]
    #         lon_point_list = [lonCorner[ii-1], lonCorner[ii-1], lonCorner[ii], lonCorner[ii]]
            
    #         # Criando um polígono para cada celula
    #         cel = Polygon(zip(lon_point_list, lat_point_list))
    #         grids.append(cel)
            
    
    

    # Abrindo shapefile com a textura do solo
    shapeSolos = gpd.read_file(inputFolder+'/Solos_5000mil/Solos_5000.shp')
    shapeSolos = shapeSolos.set_crs('epsg:4326')
    df = pd.DataFrame({'geometry':grids})
    gdf = gpd.GeoDataFrame(df, crs="EPSG:4326")
    gdf.set_geometry('geometry', inplace=True)
 
    # Recortando o tipo de solo para cada célula do domínio.
    soilIdx=[]
    for index, row in gdf.iterrows():
        clipped = gpd.clip(shapeSolos, row.geometry)
        if clipped.shape[0]>0:
            vals = []
        
            for txt in clipped['DSC_TEXTUR']:
        
                if txt == 'argilosa  ou muito argilosa':
                    vals.append(3)
                elif txt == 'media':
                    vals.append(2)
                elif txt == 'arenosa':
                    vals.append(1)
        
            if len(vals) > 0:
                soilIdx.append(pd.Series(vals).mode().iloc[0])
            else:
                soilIdx.append(np.nan)
        else:
            soilIdx.append(np.nan)

    # # Inicializando a matriz de pixels de cada idSoil no domínio
    matRegrid = np.empty((lat.shape[0]-1, lon.shape[1]-1))
    
    # fazendo o rashape
    matRegrid[:,:] = np.array(soilIdx).reshape((lat.shape[1]-1,lon.shape[0]-1)).transpose() 

    # substitui nan por 0
    matRegrid[np.isnan(matRegrid)] = 0
    
    # escreve o arquivo netCDF com a textura do solo para não precisar fazer 2 vezes
    regMap.createNETCDF(outfolder,'regridedSoilTexture_'+GDNAM,matRegrid,lon[:-1,:-1],lat[:-1,:-1])
    
    return matRegrid


def soilType_old(inputFolder,outfolder,lat,lon,D,GDNAM,dx):
    """
    função utilizada para determinar a porcentagem de particulas de um determinado
    diâmetro em cada celula do dominio. 
    
    ESTA FUNÇÃO PRECISA SER VERIFICADA 

    Parameters
    ----------
    inputFolder : path
        caminho para a pasta de inputs.
    outfolder : pat
        caminho para a pasta de outputs.
    lat : np.array
        matriz de lat do dominio.
    lon : np.array
        matriz de lon do dominio.
    D : float
        diametro da particula.
    GDNAM : str
        nome da grade de acordo com o MCIP.

    Returns
    -------
    sRef : np.array
        matriz com os valores da porcentagem de particulas com um determinado 
        diametro.

    """
    
    # abre o raster com o regrid da soiltexture
    raster_clay = nc.Dataset(outfolder+'/regridClay_'+GDNAM+'.nc', 
                        masked=True)['MAT'][:]
    raster_silt = nc.Dataset(outfolder+'/regridSilt_'+GDNAM+'.nc', 
                        masked=True)['MAT'][:]
    raster_sand = nc.Dataset(outfolder+'/regridSand_'+GDNAM+'.nc', 
                        masked=True)['MAT'][:]
    
    raster = [raster_clay,raster_silt,raster_sand]
    
    # lista com os tipos de soilTextures
    soilNames=['Clay', 'Silt', 'Sand']
        
    # inicializa a matriz que conterá os valores de porcentagem
    sRef=np.empty((len(soilNames),lat.shape[0]-1,lat.shape[1]-1))
    sRef[:,:,:] = 0
    
    # loop para cada soilTextures
    for kk,soiln in enumerate(soilNames):
        
        # abre csv com a distribuição de particulas para cada uso do solo
        # https://www.slideshare.net/slideshow/classificac3a7c3a3o-dossolosaashtosucs/49327763
        # VERIFICAR
        soilDist = pd.read_csv(inputFolder+'/tables/particleDist/'+soiln+'.csv')
        
        # Converte para micrometros os diâmetros
        xs=soilDist['D']*1000
        
        # proporção acumulada de particulas em cada diametro
        ys=soilDist['P']
        
        # função para fitar a curva logística acumulada
        f = lambda xs, a, b, e, g: b + (a - b) / (1 + (xs/e)**g)
        
        # fitando a curva
        a, b, e, g = optimize.curve_fit(f, xs, ys / 100)[0]
        
        # valores de diâmetros de 0 ao maior diametro em micrometros para usar na funçaõ fitada
        xx = np.arange(0,1000,dx)
        
        yy = f(xx,a,b,e,g)
        
        if yy[0] > 0:
            yy[0] = 0
        
        # derivada da curva acumulada, ou seja, o valor de porcentagem de um 
        # determinado diametro. 
        deriv = np.append(np.nan,np.diff(yy*100))
        
        # indice da matriz que possui o determinado diâmetro
        idx = find_nearest(xx, D)
                
        # estabelece o valor de porcentagem para um determinado diametro na matriz
        # com o mesmo tamanho do dominio
        sRef[kk,:,:]=deriv[idx]*raster[kk]/100
         
    sRef = np.sum(sRef,axis=0)
    print(sRef)
    print(sRef.max())

    return sRef

def soilType(inputFolder,outfolder,lat,lon,GDNAM,dx):
    """
    função utilizada para determinar a porcentagem de particulas de um determinado
    diâmetro em cada celula do dominio. 
    
    ESTA FUNÇÃO PRECISA SER VERIFICADA 

    Parameters
    ----------
    inputFolder : path
        caminho para a pasta de inputs.
    outfolder : pat
        caminho para a pasta de outputs.
    lat : np.array
        matriz de lat do dominio.
    lon : np.array
        matriz de lon do dominio.
    GDNAM : str
        nome da grade de acordo com o MCIP.

    Returns
    -------
    sRef : np.array
        matriz com os valores da porcentagem de particulas com um determinado 
        diametro.

    """
    
    # abre o raster com o regrid da soiltexture
    raster_clay = nc.Dataset(outfolder+'/regridClay_'+GDNAM+'.nc', 
                        masked=True)['MAT'][0,:,:]
    raster_silt = nc.Dataset(outfolder+'/regridSilt_'+GDNAM+'.nc', 
                        masked=True)['MAT'][0,:,:]
    raster_sand = nc.Dataset(outfolder+'/regridSand_'+GDNAM+'.nc', 
                        masked=True)['MAT'][0,:,:]

    MMD = [210, 125, 2]
    desv = [1.6, 1.8, 2]
    M = [raster_sand,raster_silt,raster_clay]
    
    p1 = np.arange(0.1, 1.0, 0.1)    # 0.1 a 0.9
    p2 = np.arange(1.0, 10.0, 1.0)   # 1 a 9
    p3 = np.arange(10.0, 100.0, 10) # 10 a 90
    p4 = np.arange(100.0, 1000.0, 100)
    p5 = np.arange(1000.0, 2000.0 + 1000.0, 1000.0) # 1000 a 2000
    Dp = np.concatenate([p1, p2, p3, p4, p5])
    
    pp = 2655

    ## equação https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/95JD00690

    dM_dln = []

    for j in range(0,len(MMD)):
        print(j)

        t1 = M[j][np.newaxis,:,:]/100
        t2 = np.exp((np.log(Dp[:,np.newaxis,np.newaxis])-np.log(MMD[j]))**2/(-2*np.log(desv[j])**2))
        t3 = np.sqrt(2*np.pi)*np.log(desv[j])
        
        dM = t1*t2/t3

        dM_dln.append(dM)

    dM_dln = np.nansum(dM_dln,axis=0)
    dS = dM_dln / ((2/3) * pp * Dp[:,np.newaxis,np.newaxis]**2)
    s_total = trapezoid(dS, Dp, axis=0)
    dS_rel = dS / s_total
    
    #sRef = simpson(dS_rel[np.where((Dp >= D-dx) & (Dp <= D))[0],:,:], Dp[np.where((Dp >= D-dx) & (Dp <= D))[0]], axis=0)*100

    #print(sRef)
    #print(np.nanmax(sRef))

    return dS_rel
    
def main(inputFolder,outfolder,domainShp,GDNAM,lat,lon,RESET_GRID,grids,dx):
    """
    Esta função controla a geração de arquivos de solo - claycontent, soiltexture,
    porcentagem de particulas de um determinado diametro.

    Parameters
    ----------
    inputFolder : path
        DESCRIPTION.
    outfolder : path
        DESCRIPTION.
    domainShp : geodataframe
        geodataframe com o shape do domínio.
    GDNAM : str
        nome da grade de acordo com o MCIP.
    lat : np.array
        matriz de latitudes do dominio.
    lon : np.array
        matriz de longitudes do dominio.
    D : float
        diametro da particula.
    RESET_GRID : boolean
        True ou False para resetar a matriz e reescrever o netCDF com o clay content.

    Returns
    -------
    clayRegrid : np.array
        matriz com o clay content.
    sRef : np.array
        matriz de porcentagem de particulas com um determinado diâmetro.

    """
    print('=====STARTING soilPrep.py=====' )
    
    # se existir o arquivo de regridClay para o domínio
    if (os.path.exists(outfolder+'/regridClay_'+GDNAM+'.nc')) and (os.path.exists(
            outfolder+'/regridedSoilTexture_'+GDNAM+'.nc')):
        
        # se não quiser resetar a grade = usa o regrid que já tem
        if RESET_GRID==False:
            print ('You already have the regridClay_'+GDNAM+'.nc file')
            
            # abre o netCDF com o regridClay já feito
            ds = nc.Dataset(outfolder+'/regridClay_'+GDNAM+'.nc')
            clayRegrid = ds['MAT'][:]
            ds2 = nc.Dataset(outfolder+'/regridSilt_'+GDNAM+'.nc')
            siltRegrid = ds2['MAT'][:]
            ds3 = nc.Dataset(outfolder+'/regridSand_'+GDNAM+'.nc')
            sandRegrid = ds3['MAT'][:]
            ds4 = nc.Dataset(outfolder+'/regridPB_'+GDNAM+'.nc')
            pbRegrid = ds4['MAT'][:]
            
            # abre o arquivo de regridedSoilTexture já feito
            print ('You already have the regridedSoilTexture_'+GDNAM+'.nc file')
            ds = nc.Dataset(outfolder+'/regridedSoilTexture_'+GDNAM+'.nc')
            
            # roda a função de soilType que precisa ser executada 
            # para cada diâmetro
            sRef = soilType(inputFolder,outfolder,lat,lon,GDNAM,dx)
        
        # se quiser resetar a grade    
        else:
            
          # executa a função cutSoil para cortar o arquivo original
          raster = cutSoil(domainShp,inputFolder,outfolder,GDNAM)
          
          # extraindo x e y dos pixels do raster 
          x, y = rasterLatLon(raster[0])
          
          # executa a função para fazer o regrid do clayContent
          # clayRegrid = rasterInGrid(domainShp,raster,x,y,lat,lon,grids)

          clayRegrid = rasterInGrid_optimized(
              domainShp,raster[0],x,y,lat,lon,grids)
          
          siltRegrid = rasterInGrid_optimized(
              domainShp,raster[1],x,y,lat,lon,grids)
          
          sandRegrid = rasterInGrid_optimized(
              domainShp,raster[2],x,y,lat,lon,grids)
          
          pbRegrid = rasterInGrid_optimized(
              domainShp,raster[0],x,y,lat,lon,grids)
          
          # cria o netCDF com o regrid do clayCOntent
          print('Creating netCDF')
          regMap.createNETCDF(outfolder,'regridClay_'+GDNAM,clayRegrid,lon[:-1,:-1],lat[:-1,:-1])
          regMap.createNETCDF(outfolder,'regridSilt_'+GDNAM,siltRegrid,lon[:-1,:-1],lat[:-1,:-1])
          regMap.createNETCDF(outfolder,'regridSand_'+GDNAM,sandRegrid,lon[:-1,:-1],lat[:-1,:-1])
          regMap.createNETCDF(outfolder,'regridPB_'+GDNAM,pbRegrid,lon[:-1,:-1],lat[:-1,:-1])
          
          # executa a função para fazer o regrid do soilTexture
          regridSoilTexture(outfolder,inputFolder,lat,lon,GDNAM,grids)
          
          # executa a função para estimar a porcentagem de particulas na grade
          sRef = soilType(inputFolder,outfolder,lat,lon,GDNAM,dx)
          sRef[np.isnan(sRef)] = 0  
    
    # se não existir o arquivo de regrid      
    else:
        
        # executa a função cutSoil para cortar o arquivo original
        raster = cutSoil(domainShp,inputFolder,outfolder,GDNAM)
        
        # extraindo x e y dos pixels do raster 
        x, y = rasterLatLon(raster[0])
        
        print('Clayregrid')
        
        # executa a função para fazer o regrid do clayContent
        # clayRegrid = rasterInGrid(domainShp,raster,x,y,lat,lon,grids)
        clayRegrid = rasterInGrid_optimized(
              domainShp,raster[0],x,y,lat,lon,grids)
          
        siltRegrid = rasterInGrid_optimized(
              domainShp,raster[1],x,y,lat,lon,grids)
          
        sandRegrid = rasterInGrid_optimized(
              domainShp,raster[2],x,y,lat,lon,grids)
          
        pbRegrid = rasterInGrid_optimized(
              domainShp,raster[3],x,y,lat,lon,grids)

        # cria o netCDF com o regrid do clayCOntent        
        print('Creating netCDF')
        regMap.createNETCDF(outfolder,'regridClay_'+GDNAM,clayRegrid,lon[:-1,:-1],lat[:-1,:-1])
        regMap.createNETCDF(outfolder,'regridSilt_'+GDNAM,siltRegrid,lon[:-1,:-1],lat[:-1,:-1])
        regMap.createNETCDF(outfolder,'regridSand_'+GDNAM,sandRegrid,lon[:-1,:-1],lat[:-1,:-1])
        regMap.createNETCDF(outfolder,'regridPB_'+GDNAM,pbRegrid,lon[:-1,:-1],lat[:-1,:-1])
        
        ds = nc.Dataset(outfolder+'/regridClay_'+GDNAM+'.nc')
        clayRegrid = ds['MAT'][:]
        
        # executa a função para fazer o regrid do soilTexture
        regridSoilTexture(outfolder,inputFolder,lat,lon,GDNAM,grids)
        
        # executa a função para estimar a porcentagem de particulas na grade
        sRef = soilType(inputFolder,outfolder,lat,lon,GDNAM,dx)
        sRef[np.isnan(sRef)] = 0
        
    return clayRegrid,siltRegrid,sandRegrid,pbRegrid,sRef
