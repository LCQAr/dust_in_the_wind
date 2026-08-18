#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

---------------------------metPrep.py------------------------------------------

Esta script é utilizado para determinar as condições meteorlógicas (velocidade
de fricção) e rugosidade da superfície para o domínio de modelagem. Estas condi
ções são utilizadas na estimativa da ressuspensão em cada pixel. 

Utilizei o artigo https://agupubs.onlinelibrary.wiley.com/doi/10.1002/2016MS000823
como referência. 

Ref

https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2000JD900304

Created on Mon Mar 18 13:04:17 2024

@author: leohoinaski

"""



import numpy as np
import pandas as pd
import wrf



def ustarCalc(uz,z0):
    """
    Esta função calcula a velocidade frictiva. 
    
    Parameters
    ----------
    uz : numpy array
        velocidade do vento.
        
    z0 : numpy array
        rugosidade. 
        Recebe o dado da função roughness neste mesmo script.
        
    Returns
    -------
    ustar : numpy array
        Velocidade frictiva

    """
    
    k= 0.4 # von karman constant
    
    z= 10.0 # altura de referência
    
    z0[z0<=0]=np.nan # Remove os negativos
    
    #z0[np.where(z0==0)] = np.nan
    #z0 = np.array(z0)

    #print('z0 shape: '+ str(z0.shape))
    
    #print('uz shape: '+ str(uz.shape))
    
    #print('z0 shape: '+ str(z0.shape))
    
    ustar = k*uz/np.log(z/z0) # Equação que estima a friction velocity
    
    #ustar[:,np.isnan(z0)]=np.nan # NaN para quando z0 for NaN
    
    #ustar[:,z0==0]=np.nan # NaN para quando z0 for 0
    
    #print('ustar shape: '+ str(ustar.shape))

    return ustar

def roughness(tablePath,avWRF):
    """
    
    Esta função é utilizada para calcular a rugosidade da superfície de acordo
    com o artigo https://agupubs.onlinelibrary.wiley.com/doi/10.1002/2016MS000823

    Parameters
    ----------
    tablePath : path
        Caminho para o arquivo com hs, hv e  alphas de acordo com o artigo 
        https://agupubs.onlinelibrary.wiley.com/doi/10.1002/2016MS000823
        Table 1. Monthly Vegetation Geometric Height 
        and Geometric Height and Density of Nonvegetation (Solid) 
        Elements Based on the Land Type
        
    av : numpy array
        porcentagem de área vegetada.
    al : TYPE
        porcentagem de área com o soilId.

    Returns
    -------
    z0 : np.array
        matriz com rugosidades no domínio.
    alphaV : np.array
        matriz com alphaV no domínio..
    alphaS : np.array
        matriz com alphaS no domínio..

    """
    
    # definindo altura geométrica dos sólidos e vegetação
    hs = 0.02
    hv = 0.1
    
    av = np.array(avWRF, copy=True)
    av[av>0.046]=0.046
    
    # densidade de elementos sólidos e vegetação
    lambdaS = 0.002
    lambdaV = -0.35*np.log(1-av)
    
    # calculando alpha total
    lambdaSum = lambdaS + lambdaV
    
    # estimando h conforme artigo
    h = (hv*lambdaV + hs*lambdaS)/(lambdaSum)
    
    # calculando o z0 de acordo com o artigo para as duas condições de alpha
    z0 = np.array(lambdaSum, copy=True)

    z0[lambdaSum>=0.2] = ((0.083*lambdaSum**(-0.46))*h)[lambdaSum>=0.2] 
    z0[lambdaSum<0.2] = ((0.96*lambdaSum**(1.07))*h)[lambdaSum<0.2] 
    
    return z0,lambdaV,lambdaS

def roughness_old(tablePath,avWRF):
    """
    
    Esta função é utilizada para calcular a rugosidade da superfície de acordo
    com o artigo https://agupubs.onlinelibrary.wiley.com/doi/10.1002/2016MS000823

    Parameters
    ----------
    tablePath : path
        Caminho para o arquivo com hs, hv e  alphas de acordo com o artigo 
        https://agupubs.onlinelibrary.wiley.com/doi/10.1002/2016MS000823
        Table 1. Monthly Vegetation Geometric Height 
        and Geometric Height and Density of Nonvegetation (Solid) 
        Elements Based on the Land Type
        
    av : numpy array
        porcentagem de área vegetada.
    al : TYPE
        porcentagem de área com o soilId.

    Returns
    -------
    z0 : np.array
        matriz com rugosidades no domínio.
    alphaV : np.array
        matriz com alphaV no domínio..
    alphaS : np.array
        matriz com alphaS no domínio..

    """
    
    # lendo csv com os dados de hs de acordo com o artigo
    alpS = pd.read_csv(tablePath + '/hs.csv')
    
    # lendo csv com os dados de hv de acordo com o artigo
    hv = pd.read_csv(tablePath + '/hv.csv')
    
    # estimando o alphaS com base na proporção de área vegetada e não vegetada
    alphaS = avWRF*alpS['alphaS'][0] + (1-avWRF)*alpS['alphaS'][2] 
    
    # estimando alphaV de acordo com o artigo
    alphaV = -0.35*np.log(1-avWRF)
    
    # removing inf values 
    alphaV[alphaV == np.inf] = np.nanmax(alphaV)
    alphaV[alphaV < 0] = np.nanmin(alphaV)
    
    # calculando alpha total
    alpha = alphaS + alphaV
    
    # estimando hs pela ponderação entre area vegetada e não vegetada
    hS= avWRF*alpS['hs'][0] + (1-avWRF)*alpS['hs'][2]
    
    # estimando hv pela ponderação entre area vegetada e não vegetada
    # usa a média dos meses 
    hV = avWRF*hv['mean'][0] + (1-avWRF)*hv['mean'][2]
    
    # estimando h conforme artigo
    h = (hV*alphaV + hS*alphaS)/(alphaS + alphaV)
    
    # inicializando z0
    z0 = alpha.copy() 
    # z0[alpha<0.2] = h[alpha<0.2]*0.96*alpha[alpha<0.2]**(1.07)
    # z0[alpha>=0.2] = h[alpha>=0.2]*0.083*alpha[alpha>=0.2]**(-0.46)
    
    # calculando o z0 de acordo com o artigo para as duas condições de alpha
    z0[alpha<0.2] = (0.96*alpha[alpha<0.2]**(1.07))*h[alpha<0.2]
    z0[alpha>=0.2] = (0.083*alpha[alpha>=0.2]**(-0.46))*h[alpha>=0.2]
    
    # remove nans
    # z0[:,np.isnan(al)]=np.nan
    # alphaV[:,np.isnan(al)]=np.nan
    # alphaS[:,np.isnan(al)]=np.nan
    z0 = np.array(z0)
    return z0,alphaV,alphaS

def ustarThreshold(D,clayRegrid,w,alphaV,alphaS,avWRF,z0):
    """
    Esta função calcula o limiar que define quando vai ocorrer a ressuspensão, 
    com base na velocidade frictiva ustarThreshold. 

    Parameters
    ----------
    D : int
        diâmetro da partícula.
    clayRegrid : np.array
        porporção de argila - matriz de clayContent .
    w : np.array
        matriz de umidade do solo de acordo com o output do WRF.
    alphaV : np.array
        roughness densities vegetation elements.
    alphaS : np.array
        roughness densities based on solid (nonvegetation)
    av : np.array
        matriz de proporção de área vegetada.

    Returns
    -------
    ustarT : np.array
        matriz da velocidade frictiva threshold.
    ustarTd : np.array
        matriz de ideal threshold friction velocity based on the particle size.

    """
    
    # constante (accounting for the magnitude of the interparticle cohesive forces) 
    An = 0.0123
    
    # constante (accounting for the magnitude of the interparticle cohesive forces) 
    tao = 1.65*(10**(-4)) #  kg s-2
    
    # densidade do ar
    roa = 1.227# kg/m³
    
    # densidade da particula
    # mantive esta unidade após comparar com o gráfico da figura 3 do artigo
    # https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2000JD900304
    rop = 2665 # kg/m³ assumi - montar matriz de densidades
    
    # gravidade
    g = 9.81 # m/s2
        
    # separando a equação em dois termos
    t1 = rop*g*(D*10**(-6))/roa
    t2 = tao/(roa*D*10**(-6))
    
    # equação para estimar ustarTd, conforme o artigo
    #ustarTd = np.sqrt(An*(t1+t2))
    
    gama = 1.65*(10**-4)
    d0 = 100
    b = 0.33
    sigma0 = 1
    roa = 1.227
    rop = 2665
    sigmaP = rop/roa
    g = 9.81
    
    #ustarTd = np.sqrt(An*(sigmaP*g*D*10**(-6)+(gama/(roa*D*10**(-6)))*np.exp(-sigma0*(d0/D)**b)))
    
    ustarTd = np.sqrt(An*(rop*g*(D*10**(-6))/roa+gama/(roa*D*10**(-6))))
    print(ustarTd)
    
    # Usar para teste e comparação com artigo
    #D = np.arange(0.1,1000)
    # plt.plot(D,ustarTd)
    # plt.xscale('log')
        
    # calculo da umidade do solo, conforme o artigo
    wl = (0.0014*((clayRegrid)**2)+0.17*clayRegrid)/100
    
    # Soil moisture increases the threshold friction velocity through 
    # increasing the cohesive forces suppressing the mobilization of particles.
    # inicializando a matriz de fm
    fm = np.empty(w.shape)
    fm[:,:,:]=np.nan
    fm=fm.astype(float)
    
    print("Avaliando Umidade do solo")
    #print(w.mean())
    #print(wl.mean())
    
    # estimando conforme o artigo
    fm = (1+1.21*((w-wl)**(0.68)))**(0.5)
    fm[w<wl] = 1.0 
    
    print("fator de umidade")
    #print(fm.mean())
    #print(fm.max())
    
    # constantes
    sigmaV = 1.45
    mV = 0.16
    betaV = 202
    sigmaS = 1.0
    mS = 0.5
    betaS = 90
    av = np.array(avWRF, copy=True)
    av[av>0.046]=0.046
    alphaS = 0.002
    alphaV = -0.35*np.log(1-av)
    
    # para teste
    # av = np.random.rand(1000)
    # al = 1-av
    # alphaV = -0.35*np.log(1-av)
    # tablePath = '/mnt/sdb1/windBlowDustBR/inputs/tables'
    # alpS = pd.read_csv(tablePath + '/hs.csv')
    # hv = pd.read_csv(tablePath + '/hv.csv')
    # alphaS = av*alpS['alphaS'][0] + al*alpS['alphaS'][2] 
    t1 = (1-sigmaV*mV*alphaV)
    t2 = (1+betaV*mV*alphaV)
    t3 = (1-sigmaS*mS*(alphaS/(1-av)))
    t4 = (1+betaS*mS*(alphaS/(1-av)))
    # print(t1.max())
    # print(t2.max())
    # print(t3.max())
    # print(t4.max())

    #===========================VERIFICAR!!!
    fr = (t1*t2*t3*t4)**0.5
    
    #fr =1
    
    #z0s = 3*10**-3
    
    #z0s de acordo com artigo, em função da mediana do diametro da areia https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/96JD02964
    #z0s = 435*10**-4
    
    #fr = (1-np.log(z0/z0s)/(1-np.log(0.7*(12255/z0s)**0.8)))**(-1)
    
    #fr = 1
    
    # Referência do fr = https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2010JD014649
    #fr2 = ((1-sigmaS*mS*(alphaS/(1-av)))*(1+betaS*mS*(alphaS/(1-av))))**0.5 # ref = )
    
    # Estimativa do threshold
    # ustarT=ustarTd*np.array(fm)*np.array(fr)
    ustarT=ustarTd*np.array(fm)*np.array(fr)
    # considerando apenas a correção pela umidade
    #ustarT=ustarTd*np.array(fm)
    ustarT = np.array(ustarT)
    #print(ustarT.max())
    #print("Fator de rugosidade")
    #print(fr.min())
    #print(fr.max())
    #print(ustarT)
    #print(ustarT.max())
    #print(ustarT.mean())
    
    return ustarT,ustarTd

def main(ds,tablePath,av,al,D,clayRegrid,lia,lat_index,lon_index):
    """
    Esta função controla todo o script de metPrep.

    Parameters
    ----------
    ds : netCDF object 
        objeto com o netCDF do WRF
    tablePath : path
        caminho para a pasta com as tabelas do artigo.
    av : np.array
        proporção de área vegetada.
    al : np.array
        proporção de área de cada soilId.
    D : int
        diâmetro da partícula.
    clayRegrid : np.array
        matriz com os dados de claycontent.
    lia : np.array
        vetor com datas compatíveis entre MCIP e WRF.

    Returns
    -------
    ustar : np.array
        matriz com valores de ustar para cada tempo.
    ustarT : np.array
        matriz de ustar threshold.
    ustarTd : np.array
        matriz de ustar threshold.
    av : np.array
        matriz de proporção de área vegetada do WRF.
    ustarWRF : np.array
        matriz de ustar do WRF.

    """
    print('=====STARTING metPrep.py=====' )
    
    latmesh, lonmesh = np.meshgrid(lat_index,lon_index)
    avWRF = ds['VEGFRA'][lia,lat_index,lon_index]/100
    avWRF = avWRF[:,:-1,:-1]
    ustarWRF = ds['UST'][lia,lat_index,lon_index]
    ustarWRF = ustarWRF[:,:-1,:-1]
    
    print('estimando rugosidade')
    z0,alphaV,alphaS = roughness(tablePath,avWRF)
    
    print('extraindo velocidade do vento do WRF')
    uz = np.array(wrf.g_wind.get_destag_wspd_wdir10(ds,timeidx=wrf.ALL_TIMES)[0,lia,lat_index,lon_index])
    uz = uz[:,:-1,:-1]
    
    print('calculando ustar de acordo com artigo')
    ustar = ustarCalc(uz,z0)
    
    w = ds['SMOIS'][lia,0,lat_index,lon_index]
    w = w[:,:-1,:-1]
    
    print('estimando ustarT e ustarTd de acordo com o artigo')
    ustarT,ustarTd = ustarThreshold(D,clayRegrid,w,alphaV,alphaS,avWRF,z0)
    print('u10 max = ' + str(uz.max()))
    print('u10 median = ' + str(np.median(uz)))
    print('ustarT max=' + str(ustarT.max()) )
    print('ustarT min=' + str(ustarT.min()) )
    print('ustarTd npixels=' + str(np.nansum(ustarT>0)) )
    print('ustarTd max=' + str(ustarTd))
    print('ustar max=' + str(ustar.max()))
    print('ustar mean=' + str(ustar.mean()))
    print('emissions =' + str(np.nansum(ustar>ustarT)))
    
    return ustar,ustarT,ustarTd,avWRF,ustarWRF
