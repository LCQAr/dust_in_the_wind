#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Esta classe contém as funções para a estimativa do windBlowDust 

Created on Fri Mar 15 15:20:44 2024

@author: leohoinaski
"""

import numpy as np

def wbdFlux(avWRF,alarea,sRef,clayRegrid,siltRegrid,sandRegrid,pb,ustarWRF,ustarT,ustarTd):
    
    """
    
    Esta função é utilizada para estimar o fluxo de ressuspensão do solo de 
    acordo com o artigo https://agupubs.onlinelibrary.wiley.com/doi/10.1002/2016MS000823
    
    Parameters
    ----------
    avWRF : numpy array 2D
        porcentagem com vegetação em cada pixel do WRF.
    al : numpy array 3D
        porcentagem de cada uso do solo descoberto.
    sRef : numpy array 2D
        is the relative surface area covered with particles with diameter 
        https://agupubs.onlinelibrary.wiley.com/doi/10.1002/jgrd.50313
    ustar : numpy array 2D
        Friction velocity do WRF.
    ustarT : numpy array 2D
        threshold friction velocity
    ustarTd : numpy array 2D
        threshold friction velocity based on the particle size.

    Returns
    -------
    Fdust : numpy array 3D
        total dust emission in each computational grid cell .
    
    Fhd,Fhtot,Fvtot = matrizes com fluxos horizontal, total e vertical
    
    References:
        ρb and ρp are the bulk-soil and soil particle density
        A common range of bulk density of fine-textured surface soils is 1.0–1.3 g/cm3. 
        Coarse-textured surface soils usually have bulk density in the range of 1.3–1.8 g/cm3 (Millar et al., 1965, p. 51–52).
        https://www.sciencedirect.com/topics/agricultural-and-biological-sciences/bulk-density-of-soil#:~:text=A%20common%20range%20of%20bulk,than%201%20g%2Fcm3.
        particle densities for soils range from 2.60 to 2.75 g/cm3 for mineral particles. 
        However, they can be as high as 3.0 g/cm3 for very dense particles and as 
        low as 0.9 g/ cm3 for organic particles.
        Plastic pressure, p (N/m2)	Sand: 5000 Loam: 10000 Sandy clay loam: 10000 
        Clay: 30000
        # ρb and ρp are the bulk-soil and soil particle density
        # Cα and Cβ are constant values
        # cα	0.0002∼0.001 = assume = 0.0006
        # cβ	0.63∼2.09 = assume = 1.36
        # f is the fraction of dust contained in the volume - matriz com proporções no volume
        #Plastic pressure, p (N/m2)	Sand: 5000 Loam: 10000 Sandy clay loam: 10000 Clay: 30000
        
    """
    print('=====STARTING windBlowDustCalc.py=====' )
    
    # Constantes
    # https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2010JD014649
    # Ca = 2 
    # Cb = 1.37
    g = 9.81
    roa = 1227 # g/m³
    c = 1.0
    
    # ===================CUIDADO!!!!
    # https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2010JD014649
    # p = 10000 # asumi - montar matriz de Plastic pressure com base no solo
    # rob = 1300 # kg/m³ - assumi - montar matriz de densidades
    # rop = 2600 # kg/m³ assumi - montar matriz de densidades
    
    # Usado para verificação da equação
    # clayRegrid = 0.1
    # ustar = np.linspace(0,1.4,100)
    # sRef = 0.1
    
    # foi definido com base na proporção de particulas menores que 20 micro
    # e teor de argila
    #https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2010JD014649
    #https://www.slideshare.net/slideshow/classificac3a7c3a3o-dossolosaashtosucs/49327763      
    #f = np.array(0.5*clayRegrid/100) # metade é de particulas com d<2micra
    #print('fmax = '+str(np.nanmax(f)))
    #print('fmin = '+str(np.nanmin(f)))

    # Estimando fluxo horizontal conforme o artigo   
    Fhd = ((c*roa*(ustarWRF**3))/g)*(1-(ustarT/ustarWRF))*((1+(ustarT/ustarWRF))**2)
    Fhd[ustarT>ustarWRF] = 0 # condição quando ustar não supera o limite para ressusp
    Fhd[Fhd<0] = 0
    print('Fhd max = ' + str(np.nanmax(Fhd)))
    print('Fhd npixels = ' + str(np.nansum(Fhd>0)))
    print('ustarT<ustarWRF npixels = ' + str(np.nansum(ustarT<ustarWRF)))
    
    # estimativa do fluxo horizontal total - acredito que esteja em g/ms
    Fhtot = Fhd*sRef*10**6*10**-2 # transforma para microgramas e divide a porcentagem por 100
    print('Fhtot max = ' + str(np.nanmax(Fhtot)))
    print('Fhtot npixels = ' + str(np.nansum(Fhtot>0)))
    
    #  vertical-to-horizontal dust flux ratio
    # alpha = (Ca*g*f*rob/(2*p))*(0.24+Cb*ustarWRF*np.sqrt(rop/p))
    g = 9.81 # m/s²
    
    Ca = (0.001*sandRegrid+0.0006*siltRegrid+0.0002*clayRegrid)/100 # sem unidade
    Cb = 1.37 # sem unidade
    
    p = (5000*sandRegrid+10000*siltRegrid+30000*clayRegrid)/100 #N/m²
    
    pp = 2665 #kg/m³
    #pb kg/m³
    #ustar m/s
    
    alpha = ((Ca*g*(clayRegrid/100)*pb)/(2*p))*(0.24+Cb*ustarWRF*np.sqrt(pp/p))
    print('alpha max = '+str(np.nanmax(alpha)))
    print('alpha min = '+str(np.nanmin(alpha[alpha>0])))
    
    # cálculo do fvtot - acredito que esteja em ug/m²s
    Fvtot = np.array(alpha)*Fhtot
    print('Fvtot max = ' + str(np.nanmax(Fvtot)))
    print('Fvtot npixels = ' + str(np.nansum(Fvtot>0)))
    
    # # plotagem para verificação da equação
    # fig,ax = plt.subplots(2)
    # ax[0].scatter(ustar[13,:,:],Fvtot[13,:,:]*10**6)
    # ax[1].scatter(np.nansum(alarea[:,:],axis=0),Fvtot[13,:,:]*10**6)
    # ax[0].set_yscale('log')
    # ax[1].plot(ustar,Fhtot*10**6)
    # ax[1].set_yscale('log')
    
    # calculo da emissão fluxo X area
    # inicializa a matriz
    Fdu = np.empty(Fvtot[:,:,:].shape)
    Fdu[:,:,:] = np.nan
    #Fdust = []
    print('alarea max = ' + str(np.nanmax(alarea)))
    print('alarea npixels = ' + str(np.nansum(alarea>0)))
    print(alarea.shape)
    
    alareaSum = np.nansum(alarea,axis=0)
    # loop em cada hora
    for jj in range(0,ustarWRF.shape[0]):
        Fdu[jj,:,:] = Fvtot[jj,:,:]*alareaSum

    #Fdust = alareaSum.repeat(Fvtot.shape[0]).reshape(Fvtot.shape)*Fvtot
    # transforma para numpy array
    Fdust = np.array(Fdu)/(10**6) # para gramas por segundo
    print('Fdust max = ' + str(np.nanmax(Fdust)))
    print('Fdust npixels = ' + str(np.nansum(Fdust>0)))
    # print(Fdust.shape)
    
    # soma as emissões de todos os usos do solo
    #Fdust = np.nansum(Fdust,axis=0)
    #print('Fdust max = ' + str(np.nanmax(Fdust)))
    #print(Fdust.shape)
    
    return Fdust,Fhd,Fhtot,Fvtot



