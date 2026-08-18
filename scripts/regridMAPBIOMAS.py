#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
----------------------------regridMAPBIOMAS------------------------------------

Esta classe faz as operações com o arquivo do MAPBIOMAS extraindo número de
pixels dos usos do solo sem cobertura vegetal (estabelecidos no idSoil). 





Created on Wed Mar 13 19:45:10 2024

inputs:  https://storage.googleapis.com/mapbiomas-public/initiatives/brasil/collection_8/lclu/coverage/brasil_coverage_2022.tif


https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2008JD011236


@author: leohoinaski
"""

import rasterio as rs
import rasterio.mask
import numpy as np
import os
import netCDF4 as nc
import geopandas as gpd
from shapely.geometry import Polygon
import rioxarray as riox



# def createDomainShp(wrfoutPath,lialon,lialat):
#     """
#     Esta função é utilizada para gerar o geodataframe com o domínio de modelagem
#     e extrair as coordenadas do arquivo do WRF.

#     Parameters
#     ----------
#     wrfoutPath : path
#         Caminho para o arquivo do WRF.

#     Returns
#     -------
#     domainShp : geoDataFrame
#         Geodataframe com a geometria do domínio de modelagem.
#     lat : numpy array
#         Matriz de latitudes.
#     lon : numoy array
#         matriz de longitudes.

#     """
    
#     # Abringo arquivo do WRF
#     ds = nc.Dataset(wrfoutPath)
    
#     # Extraindo latitudes e longitudes em graus
#     lat = ds['XLAT'][0,lialat,lialon]
#     lon = ds['XLONG'][0,lialat,lialon]
    
#     # Criando o retângulo do domínio
#     lat_point_list = [lat.min(), lat.max(), lat.max(), lat.min(), lat.min()]
#     lon_point_list = [lon.min(), lon.min(), lon.max(), lon.max(), lon.min()]
    
#     # Criando a geometria shapely 
#     polygon_geom = Polygon(zip(lon_point_list, lat_point_list))
    
#     # Colocando a geometria em um geodataframe
#     domainShp = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[polygon_geom])   
    
#     return  domainShp,lat,lon



def cutMapbiomas(domainShp,inputFolder,outfolder,year,GRDNAM):
    """
    Esta função é utilizada para cortar o arquivo do Mapbiomas para o domínio 
    de modelagem. Se o domínio for muito grande, ela simplesmente lê o arquivo
    original sem cortar. 
    
    FUNÇÃO NÃO ESTÁ SENDO UTILIZADA NA VERSÃO ATUAL

    Parameters
    ----------
    domainShp : geodataframe
        Geodataframe com a geometria do domínio de modelagem.
    inputFolder : path
        Caminho para a pasta de inputs.
    outfolder : path
        Caminho para a pasta de outputs.
    year : int
        Ano de referência para a modelagem.
    GRDNAM : str
        Nome da grade de modelagem de acordo com o MCIP.

    Returns
    -------
    out_meta : rasterio
        Detalhes do raster.
    arr : rasterio
        Array do raster.

    """
    
    # Abrindo o arquivo do MAPBIOMAS
    with rs.open(inputFolder+'/mapbiomas/brasil_coverage_'+str(year)+'.tif') as src:
        
        # Tenta abrir e cortar o arquivo. Se for muito grande, não cortará e passará
        # para o except
        try:
             #  Abrindo raster e cortando
            out_image, out_transform = rasterio.mask.mask(src,domainShp.geometry,
                                                          crop=True)
            # Extraindo propriedades do raster
            out_meta = src.meta
            out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})
            
            # valores iguais a 0 passam a ser 26
            out_image[out_image==0]=26
            
            # Se conseguir recortar...
            if out_meta:   
                
                # Abre um novo arquvio e salva na pasta de outputs recortado
                with rs.open(outfolder+'/mapbiomas_'+GRDNAM+'.tif', "w", **out_meta) as dest:
                    dest.write(out_image)
                    
                # Extrai a matriz de dados do arquivo recortado
                with rs.open(outfolder+'/mapbiomas_'+GRDNAM+'.tif', 'r') as ds:
                    arr = ds.read()[0,:,:] 
                    
        # Se não conseguir recortar...            
        except:
            # Extrai apenas os objetos rasterio 
            out_meta=src.meta
            arr = src 
            
    return out_meta,arr



def rasterLatLon(outfolder,GRDNAM,inputFolder,year):
    """
    Esta função é usada para extrair os arrays de x e y do raster do Mapbiomas.

    Parameters
    ----------
    outfolder : path
        Pasta com o outputs.
    GRDNAM : str
        Nome da grade de acordo com o MCIP.
    inputFolder : path
        Caminho para os inputs.
    year : int
        Ano de referência.

    Returns
    -------
    x : numpy array
        matriz de x.
    y : numpy array
        matriz de y.

    """
    
    # Se conseguir abrir o arquvio recortado... ou seja, se ele existir...
    try:
        raster = riox.open_rasterio(outfolder+'/mapbiomas_'+GRDNAM+'.tif')
        
    # Se não existir, abre o arquivo original
    except:
        raster = riox.open_rasterio(inputFolder+'/mapbiomas/brasil_coverage_'+str(year)+'.tif')
        
    # Extrai as matrizes de x e y
    x = raster.x.values
    y = raster.y.values
    
    return x, y



def rasterInGrid(x,y,lat,lon,idSoils,year,inputFolder,grids): 
    """
    Função utilizada para fazer a contagem de pixels de uma categoria de uso do
    solo na grade de modelagem. 

    Parameters
    ----------

    x : numpy array
        matriz de x do raster Mapbiomas.
    y : numpy array
        matriz de y do raster Mapbiomas.
    lat : numpy array
        matriz de lat do domínio de modelagem.
    lon : numpy array
        matriz de y do raster do domínio de modelagem.
    idSoils : int
        id do uso do solo de acordo com o Mapbiomas.
        idSoils = [23,30,25] #4.1. Praia, Duna e Areal  4.3. Mineração 4.4. Outras Áreas não Vegetadas
    year : int
        ano de referência.
    inputFolder : path
        caminho para a pasta de inputs.

    Returns
    -------
    matRegrid : numpy array
        matriz com a contagem de pixels de cada uso do solo na grade de modelagem.
    pixelsIn : numpy array
        número de pixels do Mapbiomas em cada célula do domínio de modelagem.
    av : numpy array
        proporção de pixels com área vegetada. Diferença entre usos de solo do 
        idSoils e número total de pixels na célula
    al : numpy array
        proporção de pixels com área dos idSoils
    alarea : numpy array
        área de cada um dos idSoils nas células do domínio de modelagem.

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
            
    
    # Inicializando a matriz de pixels de cada idSoil no domínio
    matRegrid=np.empty((len(idSoils),lat.shape[0]-1,lon.shape[1]-1))
    
    # loop para cada idSoil
    for kk, soilid in enumerate(idSoils):
        
        # Inciailizando número de pixels de cada idSoil no dominio
        pixelsIn =[] 
        
        # Incializando variáveis com contagem de pixels do mapbio em cada celula
        pixelInRaster=[]
        
        # loop para cada celula da grade
        for i, shape in enumerate(grids):
            
            print(str(i) +' from ' + str(len(grids)))
            
            # Abre o raster do mapbio
            with rasterio.open(inputFolder+'/mapbiomas/brasil_coverage_'+str(year)+'.tif') as src:
                
                # corta o raster usando a célula se estiver no domínio do mapbio
                try:
                    
                    #corta o raster usando a célula
                    out_image, out_transform = rasterio.mask.mask(src, [shape], crop=True)
                    
                    # Transforma em numpy
                    out_image = np.array(out_image)
                    
                    # Se o array tiver valores que não são iguais a soilid - 
                    # coloca igual a 0
                    out_image[out_image!=soilid]=0
                    
                    # Se o array tiver valores que  são iguais a soilid - 
                    # coloca igual a 1
                    out_image[out_image==soilid]=1
                    
                    # Se a soma dos valores for maior que 0 - tem soilID na celula
                    if out_image.sum()>0:
                            print('------------>Soil in grid')
                    
                    # acumula os valores de contagem de soilID nas células
                    pixelInRaster.append(out_image.sum())
                    
                    # conta o número total de pixels dentro da célula
                    pixelsIn.append(np.size(out_image))
                
                # se a célula estiver fora do domíno do arquivo do mapbio
                except:
                    print('Pixel outside raster')
                    pixelInRaster.append(np.nan)
                    pixelsIn.append(np.nan)
                    
        # preenchendo a matriz de número de pixels para cada soilID (kk)           
        matRegrid[kk,:,:] = np.array(pixelInRaster).reshape((lat.shape[1]-1,lon.shape[0]-1)).transpose() 
        
        # preenchendo a matriz de número de pixels total em cada celula
        pixelsIn = np.array(pixelsIn).reshape((lat.shape[1]-1,lon.shape[0]-1)).transpose() 
    
    # Removendo valores nan
    matRegrid[np.isnan(matRegrid)]=0
    
    # Estimando a porcentagem vegetada
    av = (pixelsIn-np.nansum(matRegrid, axis=0))/pixelsIn
    
    # Estimando a porcentagem não vegetada
    al = np.nansum(matRegrid,axis=0)/pixelsIn
    
    # Estimando a área de cada soilID 
    alarea = matRegrid*30*30
    
    return matRegrid,pixelsIn,av,al,alarea


def createNETCDF(outfolder,name,data,xlon,ylat):
    """
    

    Parameters
    ----------
    outfolder : path
        caminho para a pasta de outputs.
    name : str
        nome do arquivo de saída da função regridMAPBIOMAS.
    data : numpy array
        matriz com os dados para a escrita do netCDF.
    xlon : numpy array
        matriz de lon.
    ylat : numpy array
        matriz de lat.

    Returns
    -------
    f2 : netCDF4 object
        objecto do pacote netCDF4 criado.

    """
    
    print('===================STARTING createNETCDF.py=======================')
    
    # Criando um objeto netCDF4      
    f2 = nc.Dataset(outfolder+'/'+name+'.nc','w') #'w' stands for write 
    
    # # Specifying dimensions
    # Se o arquivo tiver 3 dimensões
    if len(data.shape)>2:
        f2.createDimension('VAR', data.shape[0])
        f2.createDimension('ROW', data.shape[1])
        f2.createDimension('COL', data.shape[2])
        
    else:
        f2.createDimension('VAR', 1)
        f2.createDimension('ROW', data.shape[0])
        f2.createDimension('COL', data.shape[1])
    
    # Building variables
    # Passing data into variables
    # matriz de lat e lon para faciliar a análise dos arquivos
    LON = f2.createVariable('LON', 'f4', ( 'VAR','ROW','COL'))
    LAT = f2.createVariable('LAT', 'f4', ( 'VAR','ROW','COL'))
    LAT[:,:] =  ylat
    LON[:,:] = xlon
    LON.units = 'degrees '
    LAT.units = 'degrees '
    
    # Criando a variável MAT que receberá os dados de cada soilID
    MAT = f2.createVariable('MAT', np.float32, ('VAR','ROW','COL'))
    if len(data.shape)>2:
        
        # loop para cada atributo de data
        for ii in range(0,data.shape[0]):
            MAT[ii,:,:] = data[ii,:,:]
            
    else:
        MAT[:,:] = data[:,:]
        
    # Fechando o arquivo netCDF
    f2.close()
    
    return f2
    


def main(GDNAM,inputFolder,outfolder,year,idSoils,RESET_GRID,
         grids,domainShp,lat,lon): 
    """
    

    Parameters
    ----------
    wrfoutPath : path
        caminho para o arquivo do WRF.
    GDNAM : str
        nome da grade de acordo com o MCIP.
    inputFolder : path
        caminho para a pasta de inputs.
    outfolder : path
        caminho para a pasta de outputs.
    year : int
        ano de referência.
    idSoils : list
        lista com os id do uso do solo que serão considerados como área não 
        vegetada e entrarão na estimativa da emissão
    RESET_GRID : str
        True or False para resetar a grade.

    Returns
    -------
    av : np.array
        proporção de área vegetada = idSoil - total
    al : np.array
        proporção de pixels com idSOil.
    alarea : np.array
        área de cada idSoil.
    lat : np.array
        matriz de lat.
    lon : np.array
        matriz de lon.
    domainShp : geodataframe
        geodataframe com o shape do domínio de modelagem.

    """
    print('=====STARTING regridMAPBIOMAS.py=====' )
    
    # Verifica se o arquivo já existe 
    if os.path.exists(outfolder+'/regridMAPBIOMAS_'+str(year)+'_'+GDNAM+'.nc'):
        
        # Se exisitr e não quiser resetar a grade, apenas abra o que já foi feito
        if RESET_GRID==False:
            print ('You already have the regridMAPBIOMAS_'+str(year)+'_'+GDNAM+'.nc file')
            
            # Cria o domínio
            #domainShp,lat,lon =  createDomainShp(wrfoutPath,lialon,lialat)
            
            # ABre o arquivo já criado
            ds = nc.Dataset(outfolder+'/regridMAPBIOMAS_'+str(year)+'_'+GDNAM+'.nc')
            
            # Abre a matriz de MAT, sendo o indice 0 = a e 1 = al e o resto é alarea
            al= ds['MAT'][0,:,:]
            av= ds['MAT'][1,:,:]
            
            # o arquivo de alarea tem 3d e a primeira dimensão é o idSoil
            alarea= ds['MAT'][2:,:,:] 
            
        #Se RESET_GRID = True ele reseta a grade mesmo que exista o arquivo
        else:
            
            # Cria o domínio
            #domainShp,lat,lon =  createDomainShp(wrfoutPath,lialon,lialat)
            
            # Extrai matriz de lat e lon
            x, y = rasterLatLon(outfolder,GDNAM,inputFolder,year)
            
            # Faz os regrids e determina número de pixels de al, av, etc
            mapbioRegrid,pixelsIn,av,al,alarea= rasterInGrid(x,y,lat,lon,idSoils,
                                                             year,inputFolder,
                                                             grids)
            
            # Esta parte serve para salvar o arquivo em netCDF para evitar 
            # rodar quando já existir o arquivo
            matRegrid2=np.empty((2+alarea.shape[0],lat.shape[0]-1,lat.shape[1]-1))
            matRegrid2[:,:,:] = np.nan
            matRegrid2[0,:,:] = al
            matRegrid2[1,:,:] = av
            matRegrid2[2:(2+alarea.shape[0]),:,:] = alarea
            
            # Cria o arquivo netCDF de regrid e coloca na pasta de out
            createNETCDF(outfolder,'regridMAPBIOMAS_'+str(year)+'_'+GDNAM,matRegrid2,lon[:-1,:-1],lat[:-1,:-1])
    
    # se não existir o arquivo...        
    else:
        
        # cria o dominio
        #domainShp,lat,lon =  createDomainShp(wrfoutPath)
        
        # cria matriz de latlon
        x, y = rasterLatLon(outfolder,GDNAM,inputFolder,year)
        
        # Faz os regrids e determina número de pixels de al, av, etc
        mapbioRegrid,pixelsIn,av,al,alarea= rasterInGrid(x,y,lat,lon,idSoils,
                                                         year,inputFolder,
                                                         grids)
        
        # Esta parte serve para salvar o arquivo em netCDF para evitar 
        # rodar quando já existir o arquivo
        matRegrid2=np.empty((2+alarea.shape[0],lat.shape[0]-1,lat.shape[1]-1))
        matRegrid2[:,:,:] = np.nan
        matRegrid2[0,:,:] = al
        matRegrid2[1,:,:] = av
        matRegrid2[2:(2+alarea.shape[0]),:,:] = alarea
        
        # Cria o arquivo netCDF de regrid e coloca na pasta de out
        createNETCDF(outfolder,'regridMAPBIOMAS_'+str(year)+'_'+GDNAM,
                     matRegrid2,lon[:-1,:-1],lat[:-1,:-1])
        
    return av,al,alarea,lat,lon,domainShp


# wrfoutPath='/media/leohoinaski/HDD/SC_2019/wrfout_d02_2019-01-03_18:00:00'
# #wrfoutPath='/mnt/sdb1/SC_2019/wrfout_d02_2019-01-01'
# GRDNAM = 'SC_2019'
# inputFolder = os.path.dirname(os.getcwd())+'/inputs'
# outfolder = os.path.dirname(os.getcwd())+'/outputs'
# year = 2021
# idSoils = [23,24,30,25] #4.1. Praia, Duna e Areal  4.2. Área Urbanizada  4.3. Mineração 4.4. Outras Áreas não Vegetadas
# matRegrid,av,lat, lon = main(wrfoutPath,GRDNAM,inputFolder,outfolder,year,idSoils)
# import matplotlib.pyplot as plt 
# rootFolder =  os.path.dirname(os.path.dirname(os.getcwd()))
# shape_path= rootFolder+'/shapefiles/BR_regions.shp'   
# borderShape = gpd.read_file(shape_path)
# fig, ax = plt.subplots()
# plt.pcolor(lon,lat,av)
# borderShape[borderShape['NM_MUN']=='Sul'].boundary.plot(edgecolor='black',linewidth=0.5,ax=ax)
