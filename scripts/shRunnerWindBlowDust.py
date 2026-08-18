#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------shRunnerWindBlowDust.py-------------------------------

Este script em python é utilizado para rodar o módulo windBlowDust e gerar os 
inputs para o CMAQ. O script usa argumentos de entrada na linha de comando do 
terminal. Os argumentos são:

   windBlowDustFolder = caminho da pasta master do módulo
   mcipPath = caminho para os arquivos do MCIP
   wrfoutFolder = caminho para os arquivos do WRF
   domain = número do domínio do WRF
   GDNAM = nome da grade conforme o MCIP
   YEAR = ano de referência da simulação
   RESET_GRID = True para resetar os arquivos intermediários gerados pelo módulo 

Created on Fri Mar 15 16:04:37 2024

Referências:
    https://agupubs.onlinelibrary.wiley.com/doi/10.1002/2016MS000823
    https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2010JD014649

@author: leohoinaski

"""
#sys.path.insert(0, rootPath)
#import regridMAPBIOMAS as regMap
#import soilPrep as sp
#import metPrep as mp
#import windBlowDustCalc as wbd
#import netCDFcreator as ncCreate
#import os
#import numpy as np
#import netCDF4 as nc
#import wrf
#import pandas as pd
#from datetime import timedelta
#import ismember
import argparse
#import windBlowDustSpeciation as wbds
import sys
import pandas as pd
import numpy as np
import geopandas as gpd
from scipy.interpolate import griddata
from shapely.geometry import Point
from scipy.integrate import simpson, cumulative_trapezoid

if __name__ == '__main__':
    
    # Argumentos de entrada
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', default=0, action='count')
    parser.add_argument('windBlowDustFolder')
    parser.add_argument('outfolder')
    parser.add_argument('mcipPath')
    parser.add_argument('wrfoutFolder')
    parser.add_argument('domain')
    parser.add_argument('GDNAM')
    parser.add_argument('YEAR', type=int)
    parser.add_argument('RESET_GRID', type=int) # 0 OU 1
    parser.add_argument('YYYYMMDD')
    args = parser.parse_args()
    contribution = []
    
    # passando os argumentos para variáveis
    windBlowDustFolder = args.windBlowDustFolder
    outfolder = args.outfolder
    mcipPath = args.mcipPath
    wrfoutFolder = args.wrfoutFolder
    domain = args.domain
    domain = 'd0'+domain
    GDNAM  = args.GDNAM
    YEAR = args.YEAR
    RESET_GRID = args.RESET_GRID
    YYYYMMDD = args.YYYYMMDD
    
    sys.path.insert(0, windBlowDustFolder+'/scripts')
    import regridMAPBIOMAS as regMap
    import soilPrep as sp
    import metPrep as mp
    import windBlowDustCalc as wbd
    import netCDFcreator as ncCreate
    import os
    import numpy as np
    import netCDF4 as nc
    import wrf
    import pandas as pd
    from datetime import timedelta
    import ismember
    #import argparse
    import windBlowDustSpeciation as wbds
    import gridDetails as grd

    for d in range(1,32):
        YYYYMMDD='2023-04-'+str(d).zfill(2)
        print(d)

        # definindo o caminho para o arquivo METCRO3D do MCIP
        mcipMETCRO3Dpath = mcipPath+'/METCRO3D_'+GDNAM+'_'+str(YYYYMMDD)+'.nc'
        mcipGRIDDOT2Dpath = mcipPath+'/GRIDDOT2D_'+GDNAM+'_'+str(YYYYMMDD)+'.nc'
        
        # definição dos caminhos das pastas de input, output e table
        inputFolder = windBlowDustFolder+'/inputs'
        tablePath = windBlowDustFolder+'/inputs/tables'
        outfolder = windBlowDustFolder+'/Outputs/'+GDNAM
        
        # condição para restar ou não os arquivos intermediários.
        if RESET_GRID == 0:
            RESET_GRID=False
        else:
            RESET_GRID=True
        
        # Dicionários de poluentes
        PM25 = {
          "Unit": '$\g.S^{-1}$',
          "tag":'PMFINE',
          "range":[0,2.5] # micrometers
        }
        
        PMC = {
          "Unit": '$\g.S^{-1}$',
          "tag":'PMC',
          "range":[2.5,10], # micrometers
          "fractions":['PMC']
        }
        
        PM10 = {
          "Unit": '$\g.S^{-1}$',
          "tag":'PM10',
          "range":[0,10] # micrometers
        }
        
        PM1 = {
          "Pollutant": "$NO_{2}$",
          "Unit": '$\g.S^{-1}$',
          "tag":'PMULTRAFINE',
          "range":[0,1] # micrometers
        }

        ALL = {
          "Unit": '$\g.S^{-1}$',
          "tag":'AllFractions',
          "fractions":['PMFINE','PMC'] # micrometers
          #"fractions":['PMFINE','PMC'] # micrometers
        }
    
        
        # Definição dos ids do MAPBIOMAS que serão utilizados na estimativa 
        # das emissões no windblowdust
        idSoils = [23,30] #4.1. Praia, Duna e Areal  4.3. Mineração 4.4. 25 Outras Áreas não Vegetadas
        
        # espaçamento entre diâmetros para integração dos valores
        dx = 2.5 # dx para integração das emissões das frações.
        
        # frações que serão calculadas
        Fractions = [PM25,PMC]  # Lista com tipo de emissão por diâmetro. 
                                #Não precisa incluir o PM10 se já tiver PM25 e PM10
        
        # condição para verificar se a pasta de output existe
        if os.path.isdir(outfolder):
            print('You have the outputs folder')
        else:
            # se não existir, cria a pasta.
            os.makedirs(outfolder, exist_ok=True)
        
        # Grid setup
        ds,datesTime,lia,domainShp,lat,lon,lat_index,lon_index,grids = grd.main(
            mcipMETCRO3Dpath,mcipGRIDDOT2Dpath,wrfoutFolder,domain)
        
        
        # executa a função de regridMAPBIOMAS
        # 2% gerando a grade, 50% com a grade já gerada
        av,al,alarea,lat,lon,domainShp = regMap.main(GDNAM,inputFolder,
                                                     outfolder,YEAR,idSoils,RESET_GRID,
                                                     grids,domainShp,lat,lon)
        
        clayRegrid,siltRegrid,sandRegrid,pbRegrid,dS_rel = sp.main(inputFolder,outfolder,domainShp,GDNAM,
                                      lat,lon,RESET_GRID,grids,dx)

        FdustTotal = []
        FhTotal = []
        FvTotal = []
        S = []

        p1 = np.arange(0.1, 1.0, 0.1)    # 0.1 a 0.9
        p2 = np.arange(1.0, 10.0, 1.0)   # 1 a 9
        p3 = np.arange(10.0, 100.0, 10) # 10 a 90
        p4 = np.arange(100.0, 1000.0, 100)
        p5 = np.arange(1000.0, 2000.0 + 1000.0, 1000.0) # 1000 a 2000
        Dp = np.concatenate([p1, p2, p3, p4, p5])
        
        S_acumulada = cumulative_trapezoid(dS_rel, Dp, axis=0, initial=0) * 100
        S_bins = np.diff(S_acumulada, axis=0)
        
        # loop para cada fração do PM
        for jj,D in enumerate(Dp[1:]):

            print(D)

            sRef = S_bins[jj]
            
            print(sRef)
            print(np.nanmax(sRef))
            # executa a função metPrep
            ustar,ustarT,ustarTd,avWRF,ustarWRF = mp.main(ds,tablePath,av,al,
                                                          D,clayRegrid,lia,
                                                          lat_index,lon_index)
            
            # executa a função windBlowDustCalc
            Fdust,Fhd,Fhtot,Fvtot = wbd.wbdFlux(avWRF,alarea,sRef,clayRegrid,
                                                siltRegrid,sandRegrid,pbRegrid,
                                                ustar,ustarT,ustarTd)
            
            # já rodou uma vez, logo, não precisa resetar os arquivos
            # intermediários
            RESET_GRID = False
            
            # acumula os valores em cada diâmetro
            FdustTotal.append(Fdust)
            FhTotal.append(Fhtot)
            FvTotal.append(Fvtot)
            S.append(sRef)
        
        # empilha os valores em um array numpy
        FdustTotal = np.stack(FdustTotal)
        FhTotal = np.stack(FhTotal)
        FvTotal = np.stack(FvTotal)
        
        # estima a massa total de particulas dentro da faixa da fração
        # faz a integral dos dados estimados
        FdustD = np.nansum(FdustTotal, axis=0)   
        Fhtot = np.nansum(FhTotal, axis=0)
        Fvtot = np.nansum(FvTotal, axis=0)
        
        S = np.nansum(S,axis=0)
        # faz a média do fluxo para cada diâmetro
        #FdustD = np.nanmedian(FdustTotal, axis=0)   
        print(FdustD.shape)
        print(np.nanmax(FdustD))
        print(S)
        print(np.nanmax(S))
        print(np.nanmin(S))
        print(S.mean())
        print(np.median(S[S>0]))
        
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors
        import geopandas as gpd
        shape_path= '/home/lcqar/BRAIN/emis/windBlowDustBR/inputs/shapefiles/BR_regions.shp'
        borderShape = gpd.read_file(shape_path)

        fig, ax = plt.subplots()
        pcm = ax.pcolor(lon[:-1,:-1],lat[:-1,:-1],S)
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
        fig.savefig(
            '/home/lcqar/BRAIN/emis/windBlowDustBR/Outputs/BR_12km/figures/sRef_'+YYYYMMDD+'.png',
            dpi=300,
            bbox_inches="tight",
            transparent=False
        )
        plt.close(fig)
        
        species_points = gpd.read_file('/home/lcqar/BRAIN/emis/windBlowDustBR/inputs/species_points/species_points.shp')
        
        classe_speciate = wbds.Speciate(grids=grids,
                              species = ['Ca','Na','K','Al','Si','Ti','Fe','As','Cd','Pb','Mn','Hg','Ni','Cl','COrg','Mg','PMother'],
                              latitude = lat,
                              longitude = lon)
        
        species_percentage = classe_speciate.create_percentage_species_with_points(
                                      species_points=species_points, 
                                      domain_shp=domainShp)
        
        
        FdustFINE = FdustD*0.07
        FdustCOARSE = FdustD*0.93
        FdustPM10 = FdustFINE+FdustCOARSE
        FdustFINESpec = classe_speciate.agg_species_emission(FdustFINE,species_percentage)
    
        # Acumula todas as estimativas de particulas sem especiação        
        FdustALL = [FdustFINE,FdustCOARSE,FdustPM10]
        FdustALL = np.stack(FdustALL)
        FdustALL = np.nansum(FdustALL, axis=0)   
        print('FdustALL    max: '+str(FdustALL.max()))
        print('FdustPMC    max: '+str(FdustCOARSE.max()))
        print('FdustPM10   max: '+str(FdustPM10.max()))
        print('FdustPMFINE max: '+str(FdustFINE.max()))
    
        # soma as emissões de cada especie no PM25
        FdustSpeciated = FdustFINESpec
        print('FdustSpeciated max: '+str(FdustSpeciated.max()))
    
        # cria o netCDF com todas as especies de particulas/frações
        ncCreate.createNETCDFtemporal(outfolder,'windBlowDust_',FdustALL,
                                      datesTime[lia],mcipMETCRO3Dpath,ALL)
    
        # cria o netCDF especiado
        ncCreate.createNETCDFtemporalSpeciated(windBlowDustFolder,outfolder,
                                               'windBlowDust_',FdustSpeciated,
                                               datesTime[lia],mcipMETCRO3Dpath,
                                               ['PCA','PNA','PK','PAL','PSI','PTI','PFE','PAS','PCD','PPB','PMN','PHG','PNI','PCL','POC','PMG','PMOTHR'])
        
        # cria o netCDF do PM10
        ncCreate.createNETCDFtemporal(outfolder,'windBlowDust_',FdustPM10,
                                          datesTime[lia],mcipMETCRO3Dpath,PM10)
        
        # cria o netCDF do PM25
        ncCreate.createNETCDFtemporal(outfolder,'windBlowDust_',FdustFINE,
                                          datesTime[lia],mcipMETCRO3Dpath,PM25)
        
        # cria o netCDF do PMC
        ncCreate.createNETCDFtemporal(outfolder,'windBlowDust_',FdustCOARSE,
                                          datesTime[lia],mcipMETCRO3Dpath,PMC)
        
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors
        import geopandas as gpd
        shape_path= '/home/lcqar/BRAIN/emis/windBlowDustBR/inputs/shapefiles/BR_regions.shp'   
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
        fig.savefig(
            '/home/lcqar/BRAIN/emis/windBlowDustBR/Outputs/BR_12km/figures/ustar_'+YYYYMMDD+'.png',
            dpi=300,
            bbox_inches="tight",
            transparent=False
        )
        plt.close(fig)
        
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
        fig.savefig(
            '/home/lcqar/BRAIN/emis/windBlowDustBR/Outputs/BR_12km/figures/ustarWRF_'+YYYYMMDD+'.png',
            dpi=300,
            bbox_inches="tight",
            transparent=False
        )
        plt.close(fig)
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
        fig.savefig(
            '/home/lcqar/BRAIN/emis/windBlowDustBR/Outputs/BR_12km/figures/avWRF_'+YYYYMMDD+'.png',
            dpi=300,
            bbox_inches="tight",
            transparent=False
        )
        plt.close(fig)
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
        fig.savefig(
            '/home/lcqar/BRAIN/emis/windBlowDustBR/Outputs/BR_12km/figures/sRef_'+YYYYMMDD+'.png',
            dpi=300,
            bbox_inches="tight",
            transparent=False
        )
        plt.close(fig)
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
        fig.savefig(
            '/home/lcqar/BRAIN/emis/windBlowDustBR/Outputs/BR_12km/figures/alarea_'+YYYYMMDD+'.png',
            dpi=300,
            bbox_inches="tight",
            transparent=False
        )
        plt.close(fig)
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
        fig.savefig(
            '/home/lcqar/BRAIN/emis/windBlowDustBR/Outputs/BR_12km/figures/clayRegrid_'+YYYYMMDD+'.png',
            dpi=300,
            bbox_inches="tight",
            transparent=False
        )
        plt.close(fig)
        
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
        fig.savefig(
            '/home/lcqar/BRAIN/emis/windBlowDustBR/Outputs/BR_12km/figures/Fvtot_'+YYYYMMDD+'.png',
            dpi=300,
            bbox_inches="tight",
            transparent=False
        )
        plt.close(fig)
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
        fig.savefig(
            '/home/lcqar/BRAIN/emis/windBlowDustBR/Outputs/BR_12km/figures/Fhtot_'+YYYYMMDD+'.png',
            dpi=300,
            bbox_inches="tight",
            transparent=False
        )
        plt.close(fig)
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
        fig.savefig(
            '/home/lcqar/BRAIN/emis/windBlowDustBR/Outputs/BR_12km/figures/FdustD_'+YYYYMMDD+'.png',
            dpi=300,
            bbox_inches="tight",
            transparent=False
        )
        plt.close(fig)
        fig, ax = plt.subplots()
        ax.scatter(ustar.flatten(),FdustD.flatten())
        ax.set_title('FdustD vs ustar')
        ax.set_yscale('log')
        #https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2010JD014649
        fig.savefig(
            '/home/lcqar/BRAIN/emis/windBlowDustBR/Outputs/BR_12km/figures/FdustDvsUstar_'+YYYYMMDD+'.png',
            dpi=300,
            bbox_inches="tight",
            transparent=False
        )
        plt.close(fig)
        fig, ax = plt.subplots()
        ax.scatter(ustar.flatten(),Fvtot.flatten())
        ax.set_title('Fvtot vs ustar')
        ax.set_yscale('log')
        #https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2010JD014649
        fig.savefig(
            '/home/lcqar/BRAIN/emis/windBlowDustBR/Outputs/BR_12km/figures/FvtotvsUstar_'+YYYYMMDD+'.png',
            dpi=300,
            bbox_inches="tight",
            transparent=False
        )
        plt.close(fig)
        
        fig, ax = plt.subplots()
        ax.scatter(ustar.flatten(),Fhtot.flatten())
        ax.set_title('Fhtot vs ustar')
        ax.set_yscale('log')
        fig.savefig(
            '/home/lcqar/BRAIN/emis/windBlowDustBR/Outputs/BR_12km/figures/FhtotvsUstar_'+YYYYMMDD+'.png',
            dpi=300,
            bbox_inches="tight",
            transparent=False
        )
        plt.close(fig)
        fig, ax = plt.subplots()
        ax.scatter(alarea[1,:,:].repeat(Fvtot.shape[0]).flatten(),Fvtot.flatten())
        ax.set_title('Fvtot vs alarea')
        ax.set_yscale('log')
        fig.savefig(
            '/home/lcqar/BRAIN/emis/windBlowDustBR/Outputs/BR_12km/figures/Fvtotvsalarea_'+YYYYMMDD+'.png',
            dpi=300,
            bbox_inches="tight",
            transparent=False
        )
        plt.close(fig)
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
        fig.savefig(
            '/home/lcqar/BRAIN/emis/windBlowDustBR/Outputs/BR_12km/figures/PMFINE_'+YYYYMMDD+'.png',
            dpi=300,
            bbox_inches="tight",
            transparent=False
        )
        plt.close(fig)
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
        
        fig.savefig(
            '/home/lcqar/BRAIN/emis/windBlowDustBR/Outputs/BR_12km/figures/PMC_'+YYYYMMDD+'.png',
            dpi=300,
            bbox_inches="tight",
            transparent=False
        )
        plt.close(fig)
