#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 15:08:54 2024

-------------------------windBlowDustSpeciation.py-----------------------------

classe utilizada para fazer a especiação química das emissões do windblowdust. 
Utilizamos o speciate para elabora a planilha weigth_perc_PM_CMAQ.csv com a 
especiação do material particulado emitido pelo solo.  


@author: leohoinaski

"""


import pandas as pd
import numpy as np
import geopandas as gpd
from scipy.interpolate import griddata
from shapely.geometry import Point

def contribution_areas(windBlowDustFolder,grids,lat,lon):
    
    # abrindo o shp com regioes com ferro
    shp_mineracao = gpd.read_file(windBlowDustFolder+'/inputs/MiningBR/BRASIL.shp')

    shp_mineracao = shp_mineracao[shp_mineracao['FASE'].isin(
        ['CONCESSÃO DE LAVRA','LAVRA GARIMPEIRA','LICENCIAMENTO','REGISTRO DE EXTRAÇÃO']
        )]
         
    shp_iron = shp_mineracao[shp_mineracao['SUBS'].isin(['FERRO','MINÉRIO DE FERRO'])]
    shp_iron = gpd.GeoDataFrame(geometry=[shp_iron.unary_union], crs=shp_iron.crs)
    
    # cria um gdf com os grids
    df = pd.DataFrame({'geometry':grids})
    gdf = gpd.GeoDataFrame(df, crs=shp_iron.crs)
    gdf.set_geometry('geometry', inplace=True)
    
    # calcula a porcentagem da área coberta
    gdf['prct_iron'] = gdf.geometry.intersection(shp_iron.geometry.iloc[0]).area/ gdf.geometry.area
    
    array_iron = gdf['prct_iron'].to_numpy().reshape((lat.shape[1]-1,lon.shape[0]-1)).transpose()

    contribution = np.zeros((2,lat.shape[0]-1, lon.shape[1]-1))
    
    contribution[0,:,:] = 1-array_iron

    contribution[1,:,:] = array_iron
    
    return contribution

def speciate(windBlowDustFolder,FdustD,grids,lat,lon,contribution):
    """
    função para a especiação química das emissões do windblowdust

    Parameters
    ----------
    windBlowDustFolder : path
        caminho para a pasta do módulo windblowdust.
    FdustD : np.array
        matriz com as emissões de partículas

    Returns
    -------
    FdustDNew : np.array
        matriz com as emissões especiadas.
    """
   
    
    print('=====STARTING windBlowDustSpeciation.py=====' )
    # abrindo csv com os perfis de especiação 
    spc = pd.read_csv(windBlowDustFolder+'/inputs/tables/weigth_perc_PM_All_CMAQ.csv')
    
    # usa todas as linhas que não tiver null
    spc = spc[~spc['SPECIES_NAME'].isnull()]
    
    # inicializa a matriz com as emissões especiadas
    FdustDNew = np.zeros([FdustD.shape[0],spc.shape[0],FdustD.shape[1],FdustD.shape[2]])
    
    if type(contribution) == list:
        contribution = contribution_areas(windBlowDustFolder,grids,lat,lon)
    
    # loop para cada espécie
    for index, row in spc.iterrows():
        lista_tipos=[]
        for type_mine in range(contribution.shape[0]):

            lista_tipos.append(contribution[type_mine,:,:]*FdustD*(row[type_mine+7]/100))

        FdustDNew[:,index,:,:] = np.nansum(np.stack(lista_tipos), axis=0)

    return FdustDNew, contribution


'''

def contribution_areas(windBlowDustFolder,grids,lat,lon):
    
    # abrindo o shp com regioes com ferro
    shp_mineracao = gpd.read_file(windBlowDustFolder+'/inputs/MiningBR/BRASIL.shp')
    shp_mineracao = shp_mineracao[shp_mineracao['FASE'].isin(
        ['CONCESSÃO DE LAVRA','LAVRA GARIMPEIRA','LICENCIAMENTO','REGISTRO DE EXTRAÇÃO']
        )]
    
    tipos_minas = [['FERRO','MINÉRIO DE FERRO'],['MINÉRIO DE OURO','OURO']]
    
    lista_minas = []
    
    for k, tipo in enumerate(tipos_minas):
    
        shp_mina = shp_mineracao[shp_mineracao['SUBS'].isin(tipo)]
        shp_mina = gpd.GeoDataFrame(geometry=[shp_mina.unary_union], crs=shp_mina.crs)
        
        shp_mina_sem_sobreposicao = shp_mina.overlay(shp_mineracao[~shp_mineracao['SUBS'].isin(tipo)], how='difference')
        
        shp_mina_com_sobreposicao = shp_mina.overlay(shp_mineracao[~shp_mineracao['SUBS'].isin(tipo)], how='intersection')
        
        shp_mina_com_sobreposicao = shp_mina_com_sobreposicao.groupby('geometry').agg({
            'SUBS': lambda x: list(set(x)),
        }).reset_index()
        
        shp_mina_com_sobreposicao = gpd.GeoDataFrame(
            shp_mina_com_sobreposicao,
            geometry='geometry',
            crs=shp_mina.crs
        )
        
        if k == 0:
            # cria um gdf com os grids
            df = pd.DataFrame({'geometry':grids})
            gdf = gpd.GeoDataFrame(df, crs=shp_mina.crs)
            gdf.set_geometry('geometry', inplace=True)
        
        gdf['prct_sem_sob'] = gdf.geometry.intersection(shp_mina_sem_sobreposicao.geometry.iloc[0]).area/ gdf.geometry.area
        
        gdf['prct_com_sob'] = gdf.geometry.intersection(shp_mina_com_sobreposicao.geometry.iloc[0]).area*0.5/ gdf.geometry.area
        
        gdf['prct'] = gdf['prct_sem_sob']+gdf['prct_com_sob'] 
        
        lista_minas.append(gdf['prct'].to_numpy().reshape((lat.shape[1]-1,lon.shape[0]-1)).transpose())
    
    contribution = np.zeros((len(lista_minas)+1,lat.shape[0]-1, lon.shape[1]-1))
    
    contribution[0,:,:] = 1-np.nansum(lista_minas[:], axis=0)
    
    for i in range(len(lista_minas)):
            
        contribution[i+1,:,:] = lista_minas[i]
    
    return contribution

def speciate(windBlowDustFolder,FdustD,grids,lat,lon,contribution):
    """
    função para a especiação química das emissões do windblowdust

    Parameters
    ----------
    windBlowDustFolder : path
        caminho para a pasta do módulo windblowdust.
    FdustD : np.array
        matriz com as emissões de partículas

    Returns
    -------
    FdustDNew : np.array
        matriz com as emissões especiadas.

    """
    
    print('=====STARTING windBlowDustSpeciation.py=====' )
    # abrindo csv com os perfis de especiação 
    spc = pd.read_csv(windBlowDustFolder+'/inputs/tables/weigth_perc_PM_All_CMAQ.csv')
    
    # usa todas as linhas que não tiver null
    spc = spc[~spc['SPECIES_NAME'].isnull()]
    
    # inicializa a matriz com as emissões especiadas
    FdustDNew = np.zeros([FdustD.shape[0],spc.shape[0],FdustD.shape[1],FdustD.shape[2]]) 
    
    if type(contribution) == list:
        contribution = contribution_areas(windBlowDustFolder,grids,lat,lon)
    
    # loop para cada espécie
    for index, row in spc.iterrows():
        
        lista_tipos = []
        
        for k in range(contribution.shape[0]):
            
            print(k)
            
            lista_tipos.append(contribution[k,:,:]*FdustD*(row[k+7]/100))                                 
            
        FdustDNew[:,index,:,:] = np.nansum(np.stack(lista_tipos), axis=0)  
        
    return FdustDNew, contribution

'''

#%%

class Speciate:
    
    def __init__(
            self,
            species: list = None,
            grids: list = None,
            latitude: np.ndarray = None,
            longitude:np.ndarray = None,
            species_profile:list = [],
            species_percentage:list = None,
            fdust:np.ndarray = None
        ):
        '''
        Inicializa o objeto da classe Speciate

        Parameters
        ----------
        species : list, optional
            Lista do nome das espécies químicas que será usado como cabeçalho.
            The default is None.
        grids : list, optional
            Lista dos polígonos da grade. The default is None.
        latitude : np.ndarray, optional
            Array 2D da latitude de cada célula. The default is None.
        longitude : np.ndarray, optional
            Array 2D da longitude de cada célula. The default is None.
        species_profile : list, optional
            Lista do peso de cada espécie. The default is None.
        species_percentage : np.ndarray, optional
            Array 3D do peso de cada espécie nas células do domínio. The default is None.
        fdust : np.ndarray, optional
            Array 3D da emissão de MP para cada hora nas células do domínio. The default is None.

        Returns
        -------
        None.

        '''

        self.fdust = fdust
        self.species = species
        self.species_profile = species_profile
        self.species_percentage = species_percentage
        self.grids = grids
        self.lat = latitude
        self.lon = longitude
        
    def __repr__(self):
        return 'MeuObjeto'
        
    def __rectify_species_list(self, profile_soil_species:list, soil_species:list):
        '''
        Corrige a lista de espécies que será adicionada para ficar no mesmo formato
        do array

        Parameters
        ----------
        profile_soil_species : list
            Lista da porcentagem de cada espécie
        soil_species : list
            Lista das espécies nas células do domínio

        Returns
        -------
        list
            Lista da porcentagem de cada espécie adequada para o objeto

        '''
        
        # Cria dataframe com espécies e perfil do solo que atualizará
        soil_df = pd.DataFrame({
                'profile': profile_soil_species,
                'species': soil_species})
        
        # Cria dataframe com as espécies do objeto
        species_df = pd.DataFrame({
                'species': self.species})
        
        # Junta os dois dataframes e preenche com 0 onde não há valor do perfil
        species_df = pd.merge(species_df, soil_df, on='species', how='left').fillna(0)
        
        return species_df['profile']
        
    def create_percentage_species(self):
        '''
        Cria o array com o peso de cada espécie no domínio a partir 
        das espécies, perfil e domínio de entrada 

        Returns
        -------
        np.ndarray
            Array 3D do peso de cada espécie nas células do domínio.

        '''
        
        # Cria o array com o peso de cada espécie
        self.species_percentage = (np.array(self.species_profile)[:,np.newaxis,np.newaxis] * # Cria um array 3D com os valores de peso de cada espécie
                                   np.ones(len(self.grids)) # Multiplica por um array do tamanho da grade com valor 1
                                   .reshape((self.lat.shape[1]-1,self.lon.shape[0]-1)) # Reformata os valores para 2D com lat e lon
                                   .transpose()[np.newaxis,:,:]) # Faz a transposição do array
        
        return self.species_percentage
    
    def create_percentage_species_with_points(self, species_points:gpd.GeoDataFrame, domain_shp:gpd.GeoDataFrame):
        '''
        

        Parameters
        ----------
        species_points : gpd.GeoDataFrame
            shape com os pontos e valores respectivos para cada espécie em
            cada ponto.
        domain_shp : gpd.GeoDataFrame
            shape do domínio.

        Returns
        -------
        np.ndarray
            Array 3D do peso de cada espécie nas células do domínio.

        '''
        
        # Cria uma lista com as espécies e geometria e retira PMOthr
        species_and_geometry = self.species + ['geometry']
        species_and_geometry.remove('PMother')
        
        # Filtra species_points pela lista de species e geometria
        species_points = species_points[species_and_geometry]
        
        # Define z_grid como array 3D de valores zerados e primeira dimensão do tamanho da lista species
        z_grid = np.zeros([len(self.species),self.lat.shape[0]-1,self.lon.shape[1]-1])
        
        # Cria o x_grid e y_grid
        bounds = domain_shp.total_bounds  
        grid_x = np.linspace(bounds[0], bounds[2], self.lon.shape[1]-1)
        grid_y = np.linspace(bounds[1], bounds[3], self.lat.shape[0]-1)
        x_grid, y_grid = np.meshgrid(grid_x, grid_y)
        
        # Remove geometry da lista
        species_and_geometry.remove('geometry')
        
        for index, content in enumerate(species_and_geometry):
            
            # Retira o x, y e z do species_points
            x = species_points.dropna(subset=[content,'geometry']).geometry.x.values
            y = species_points.dropna(subset=[content,'geometry']).geometry.y.values
            z = species_points.dropna(subset=[content,'geometry'])[content].values 
            
            # Faz a interpolação linear entre os pontos
            z_linear = griddata((y, x), z, (y_grid, x_grid), method='linear')
            
            # Faz a interpolação nearest para preencher os NaNs
            z_nearest = griddata((y, x), z, (y_grid, x_grid), method='nearest')
            
            # Faz a substituição dos NaNs do linear pelo nearest
            z_grid[index,:,:] = np.where(np.isnan(z_linear), z_nearest, z_linear)
            print(z_grid.shape)
        
        z_grid[-1,:,:] = 1 - sum(z_grid[:-1,:,:])
        
        z_grid[-1,:,:] = z_grid[-1,:,:][z_grid[-1,:,:]<0] = 0
        
        z_grid = z_grid/sum(z_grid)
        
        # Define o z_grid como species_percentage
        self.species_percentage = z_grid
    
        return self.species_percentage
    
    def update_percentage_species(self, soil_shp:gpd.GeoDataFrame, 
                                  profile_soil_species:list, soil_species:list):
        '''
        Função para atualizar o species_percentage com um novo uso do solo e seu respectivo perfil

        Parameters
        ----------
        soil_shp : gpd.GeoDataFrame()
            shape do solo
        profile_soil_species : list
            lista da porcentagem de cada espécie para o tipo de solo
        soil_species: list
            lista das espécies para o tipo de solo

        Returns
        -------
        np.ndarray
            Array 3D do peso de cada espécie nas células do domínio.

        '''
        
        # Adiciona o PMOthr para fechar 100% da especiação
        soil_species = soil_species + ['PMother']
        profile_soil_species = profile_soil_species + [1-np.sum(profile_soil_species)]
        
        # Roda a função para corrigir a lista de espécies
        profile_soil_species = self.__rectify_species_list(profile_soil_species,soil_species)
        
        # Criar intesecção com grids
        cells_shp = gpd.GeoDataFrame(geometry=self.grids, crs='epsg:4674')
        
        # Realizar sjoin para gastar menos memória
        candidates = gpd.sjoin(cells_shp,soil_shp,how='inner',predicate='intersects')
                
        # Fazer intersecção
        cells_shp_intersect = cells_shp.geometry.loc[candidates.index]
        soil_shp_intersect = soil_shp.geometry.loc[candidates['index_right']]
        intersection_areas = cells_shp_intersect.intersection(soil_shp_intersect,align=False).area
        intersection_areas = intersection_areas.groupby(intersection_areas.index).sum()
        
        # Criar coluna de porcentagem do uso de determinado uso do solo
        cells_shp = pd.concat([cells_shp, intersection_areas.rename('percentage')], axis=1).fillna(0)
        cells_shp['percentage'] = cells_shp['percentage']/cells_shp.geometry.area
        percentage_soil_domain = cells_shp['percentage'].to_numpy().reshape((self.lat.shape[1]-1,self.lon.shape[0]-1)).transpose()    
        
        # Atualizar o self.species_percentage com os novos valores de array, 
        # multiplicando pela proporção do valor e somando entre si
        self.species_percentage = (self.species_percentage *
                                   (1-percentage_soil_domain)[np.newaxis,:,:] +
                                    np.array(profile_soil_species)[:,np.newaxis,np.newaxis] *
                                    percentage_soil_domain[np.newaxis,:,:])
        
        return self.species_percentage
    
    def agg_species_emission(self, fdust:np.ndarray = None, species_percentage:np.ndarray = None):
        '''
        Multiplica a grade de porcentagem das espécies com a grade de emissão por hora.

        Parameters
        ----------
        fdust : np.ndarray, optional
            Array 3D da emissão de MP para cada hora nas células do domínio. The default is None.
        species_percentage : np.ndarray, optional
            Array 3D do peso de cada espécie nas células do domínio. The default is None.
            
        Returns
        -------
        np.ndarray
            Array 4D das emissões de MP especiadas.

        '''
        
        # if e else para criar array com a porcentagem das espécies ou igualar a que foi dado de entrada
        if self.species_percentage is None and species_percentage is None:
            self.create_percentage_species()
        elif species_percentage is not None:
            self.species_percentage = species_percentage
        
        if fdust is not None:
            self.fdust = fdust

        # Multiplica as emissões com a grade de porcentagem de poluentes (np.newaxis é equivalente a None)
        return np.multiply(self.fdust[:,np.newaxis,:,:],
                           self.species_percentage[np.newaxis,:,:,:])
    
