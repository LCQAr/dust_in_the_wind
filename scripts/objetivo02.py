from netCDF4 import Dataset
from matplotlib.colors import LogNorm, ListedColormap
import contextily as cx
import pyproj
import os
import numpy as np
import xarray as xr
import glob
import matplotlib.pyplot as plt
import os
import xarray as xr
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
from shapely.geometry import box
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import shutil
import netCDF4 as nc
from matplotlib.colors import LinearSegmentedColormap, Normalize
import pickle
import textwrap
import regridMAPBIOMAS as regMap
import gridDetails as grd

###########################################################################

meses = ['01','02','03','04','05','06','07','08','09','10','11','12']

path = "/home/lcqar/BRAIN/emis/windBlowDustBR/Outputs/BR_12km/"

###########################################################################

def agg_wbd(path,op):

    final_dict = {
        "PM10":{"yearly":pd.DataFrame(),"monthly":{},"hourly":{}},
        "PMC":{"yearly":pd.DataFrame(),"monthly":{},"hourly":{}},
        "PMFINE":{"yearly":pd.DataFrame(),"monthly":{},"hourly":{}}}
    
    meses = ['01','02','03','04','05','06','07','08','09','10','11','12']
    
    for mes in meses:
        print(mes)
        pasta = path
        
        for file in os.listdir(pasta):
            
            for var in ["PM10","PMC","PMFINE"]:
                
                if file.startswith("windBlowDust_"+var+"_2023-"+mes):
                    
                    print("Verificando "+var+" do arquivo " + file)

                    dir_data = os.path.join(pasta,file)
                    data = xr.open_dataset(dir_data)

                    tflag = data['TFLAG'].values
                    tflag = tflag[:, 0, :]
                    ano = tflag[:,0] // 1000
                    dia_juliano = tflag[:,0]%1000
                    horas = tflag[:,1] //10000

                    df = pd.DataFrame({
                        'year':ano,
                        'day_of_year':dia_juliano,
                        'hour':horas})

                    time = pd.to_datetime(df['year'].astype(str) + df['day_of_year'].astype(str).str.zfill(3), format='%Y%j')

                    for freq in ['yearly','monthly','hourly']:

                        if freq == 'monthly':
                            time_group = time.dt.month 
                            unique_times = np.unique(time_group)
                        
                        elif freq == 'weekly':
                            time_group = time.dt.weekday
                            unique_times = np.arange(7)
                        
                        elif freq == 'hourly':
                            time_group = time.dt.hour
                            unique_times = np.arange(24)
                        
                        elif freq == 'yearly':
                            time_group = None
                        
                        if freq != 'yearly':
                            for t in unique_times:
                                print(t)
                                if freq == 'hourly':
                                    time_indices = np.where(unique_times == t)[0]
                                else:
                                    time_indices = np.where(time_group == t)[0]
                                
                                print(time_indices)
                                
                                time_indices = time_indices[time_indices != 24]
                                
                                print(time_indices)
                                
                                #Op in the time inverval
                                pol_2d = pd.DataFrame(
                                    op(np.array(data[var][time_indices, :, :]), axis=0).flatten()
                                    ).rename(columns={0: 'wbd'})
                                
                                max_value = pol_2d.max()
                                print(f'max value of {var} in the tspep {t} for {freq} = {max_value}')
                                
                                #Add values for all sectors in the df
                                if t not in final_dict[var][freq]:
                                    final_dict[var][freq][t] = pol_2d
                                else:
                                    final_dict[var][freq][t] = pd.concat([final_dict[var][freq][t], pol_2d], axis=1)
                        else:
                            # Op with emissions for entire year
                            time_indices = range(len(time))
                            pol_2d = pd.DataFrame(
                                op(np.array(data[var][time_indices, :, :]), axis=0).flatten()
                                ).rename(columns={0: 'wbd'})
                            max_value = pol_2d.max()
                            print(f'max value of {var} for {freq} = {max_value}')
                            # Add the values to final df (yearly)
                            if final_dict[var][freq].empty:
                                final_dict[var][freq] = pol_2d
                            else:
                                final_dict[var][freq] = pd.concat([final_dict[var][freq], pol_2d], axis=1)

    return final_dict

def agg_temporal(final_dict,op):

    for var in ['PM10','PMC','PMFINE']:
        
        for freq in ['yearly','monthly','hourly']:

            if freq == 'monthly':
                unique_times = np.arange(12)
            
            elif freq == 'weekly':
                unique_times = np.arange(7)
            
            elif freq == 'hourly':
                unique_times = np.arange(24)
            
            if freq != 'yearly':
            
                for t in unique_times:
                    
                    # Renomear colunas repetidas para evitar sobrescrita
                    final_dict[var][freq][t].columns = pd.Index([f"{col}_{i}" for i, col in enumerate(final_dict[var][freq][t].columns)])
                    
                    # Criar um dicionário para agrupar colunas pelo prefixo (antes do "_")
                    groups = {}
                    for col in final_dict[var][freq][t].columns:
                        base_name = col.split("_")[0]  # Pega o nome original da coluna
                        groups.setdefault(base_name, []).append(col)
                    
                    # Calcular a média para cada grupo de colunas
                    final_dict[var][freq][t] = pd.DataFrame({col: op(final_dict[var][freq][t][cols], axis=1) for col, cols in groups.items()})
                
            else:
                # Renomear colunas repetidas para evitar sobrescrita
                final_dict[var][freq].columns = pd.Index([f"{col}_{i}" for i, col in enumerate(final_dict[var][freq].columns)])
                
                # Criar um dicionário para agrupar colunas pelo prefixo (antes do "_")
                groups = {}
                for col in final_dict[var][freq].columns:
                    base_name = col.split("_")[0]  # Pega o nome original da coluna
                    groups.setdefault(base_name, []).append(col)
                
                # Calcular a média para cada grupo de colunas
                final_dict[var][freq] = pd.DataFrame({col: op(final_dict[var][freq][cols], axis=1) for col, cols in groups.items()})
    
    return final_dict

def plot_temporal():
    a = 1
    return a

def plot_max_min():
    a = 1
    return a

def markhen_index():
    a = 1
    return a

def plot_box_plot(dados_boxplot,labels,nome_fig,escala,label_x,label_y,eixo,outliers,size,estados_br):

    # Criação do boxplot
    fig,ax = plt.subplots(figsize=size)

    ax.boxplot(
        dados_boxplot,
        labels=labels,
        showfliers= outliers, # mostra outliers
        vert = eixo
    )

    if estados_br == True:
        regioes = {
        'Sul': {'cor': '#ccece6','estados': ['RS', 'SC', 'PR']},
        'Sudeste': {'cor': '#fddbc7','estados': ['SP', 'MG', 'RJ', 'ES']},
        'Centro-Oeste': {'cor': '#d1e5f0','estados': ['MT', 'MS', 'GO', 'DF']},
        'Nordeste': {'cor': '#fde0ef','estados': ['BA', 'SE', 'AL', 'PE', 'PB', 'RN', 'CE', 'PI', 'MA']},
        'Norte': {'cor': '#e6f5c9','estados': ['TO', 'PA', 'AP', 'RR', 'AM', 'AC', 'RO']}}
        
        posicoes = {uf: i+1 for i, uf in enumerate(labels)}

        faixas = []

        for regiao, dados in regioes.items():
            idx = [posicoes[uf] for uf in dados['estados'] if uf in posicoes]

            if idx:
                inicio = min(idx) - 0.5
                fim = max(idx) + 0.5

                faixas.append({
                    'regiao': regiao,
                    'inicio': inicio,
                    'fim': fim,
                    'cor': dados['cor']
                })

        for f in faixas:

            if eixo == False:  # horizontal
                ax.axvspan(
                    f['inicio'],
                    f['fim'],
                    color=f['cor'],
                    alpha=1,
                    zorder=0
                )

            else:  # vertical
                ax.axhspan(
                    f['inicio'],
                    f['fim'],
                    color=f['cor'],
                    alpha=1,
                    zorder=0
                )

    if eixo == False:
        ax.set_xscale(escala)
    else:
        ax.set_yscale(escala)

    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)

    ax.grid(False, alpha=0.3)

    plt.tight_layout()

    plt.savefig('/home/lcqar/BRAIN/emis/windBlowDustBR/Outputs/BR_12km/objetivo02/boxplot_' + nome_fig, dpi=300, bbox_inches="tight")

    plt.close()

def plot_box_plot_ano_estado(dfs,lat,lon,shp,var):
    # Cria um dataframe
    df = pd.DataFrame({
        'longitude': lon, # Adiciona uma coluna com longitude
        'latitude': lat # Adiciona uma coluna com latitude
    })

    # Cria a coluna geometria a partir de longitude e latitude no DataFrame
    df["geometry"] = df.apply(lambda row: Point(row["longitude"], row["latitude"]), axis=1)

    # Converte o df para GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

    gdf = gdf.to_crs("EPSG:31983")

    gdf['geometry'] = gdf.geometry.apply(lambda point: point_to_square(point,5750))

    gdf = gdf.to_crs("EPSG:4674")

    dados_boxplot = []
    labels_estado = ['RS','SC','PR','SP','MG','RJ','ES','BA','SE','AL','PE','PB','RN','CE','PI','MA','TO','PA','AP','RR','AM','AC','RO','MT','MS','GO','DF']
    size = (14,6)
    labels = []

    for estado in labels_estado:

        shp_estado = shp[shp["SIGLA_UF"]==estado]

        df = dfs[var]['yearly']

        df['geometry'] = gdf.loc[df.index, "geometry"]

        df = gpd.GeoDataFrame(df,geometry="geometry",crs=gdf.crs)

        df = gpd.clip(df, shp_estado)

        df = df.drop(columns=['geometry'])

        #print(df)

        # Transforma em array 1D
        valores = df.values.flatten()

        #print('Valores são')
        #print(valores)

        # Filtra valores maiores que 0
        valores = valores[valores > 0]

        #print('Valores sem 0 são')
        #print(valores)

        print(estado)
        print(len(valores))
        labels.append(estado+' (n = ' + str(len(valores)) + ')')

        dados_boxplot.append(valores)

    nome_fig = var+'_anual_BR'
    escala = 'log'
    label_x = 'Valor (g.ano/s)'
    label_y = 'Estados'
    eixo = False
    outliers = True
    size = (6,14)
    estados_br = True

    plot_box_plot(dados_boxplot,labels,nome_fig,escala,label_x,label_y,eixo,outliers,size,estados_br)

def plot_box_plot_ano_solo(dfs,lat,lon,shp,var):
    ds,datesTime,lia,domainShp,lat,lon,lat_index,lon_index,grids = grd.main(
        '/home/lcqar/GAR_BR/mcip/BR_12km/METCRO3D_BR_12km_2023-09-27.nc',
        '/home/lcqar/GAR_BR/mcip/BR_12km/GRIDDOT2D_BR_12km_2023-09-27.nc',
        '/home/lcqar/GAR_BR/WRF/2023/2023_09','d02')
    
    av,al,alarea,lat,lon,domainShp = regMap.main('BR_12km','inputFolder',
                                                 '/home/lcqar/BRAIN/emis/windBlowDustBR/Outputs/BR_12km',
                                                 2023,[23,30],False,
                                                 grids,domainShp,lat,lon)
    
    content_dune = (alarea[0,:,:]/(alarea[0,:,:]+alarea[1,:,:])).flatten()
    content_mine = (alarea[1,:,:]/(alarea[0,:,:]+alarea[1,:,:])).flatten()
    
    print(content_dune.max())
    print(content_mine.max())

    for soil in ['dune','mine']:

        if soil == 'dune':
            content=content_dune
        elif soil == 'mine':
            content=content_mine

        for freq in ['monthly', 'hourly']:

            # Definição dos rótulos
            if freq == 'monthly':
                labels = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
                          'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']

            elif freq == 'weekly':
                labels = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sab', 'Dom']

            elif freq == 'hourly':
                labels = list(np.arange(24))

            dados_boxplot = []

            # Percorre cada DataFrame daquela frequência
            for i in dfs[var][freq]:

                df = dfs[var][freq][i]

                valores = df.values.flatten()

                valores = valores*content

                valores = valores[valores > 0]

                dados_boxplot.append(valores)

            nome_fig = var+'_'+freq+'_'+soil
            escala = 'log'
            label_x = 'Período'
            label_y = 'Valores'
            eixo = True
            outliers = False
            size = (14,6)
            estados_br = False

            plot_box_plot(dados_boxplot,labels,nome_fig,escala,label_x,label_y,eixo,outliers,size,estados_br)

def plot_box_plot_brasil(dfs,shp,var):

    for freq in ['monthly', 'hourly']:

        # Definição dos rótulos
        if freq == 'monthly':
            labels = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
                      'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']

        elif freq == 'weekly':
            labels = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sab', 'Dom']

        elif freq == 'hourly':
            labels = list(np.arange(24))

        dados_boxplot = []

        # Percorre cada DataFrame daquela frequência
        for i in dfs[var][freq]:

            df = dfs[var][freq][i]

            valores = df.values.flatten()
            print(valores)

            valores = valores[valores > 0]

            dados_boxplot.append(valores)

        nome_fig = var+'_'+freq+'_BR'
        escala = 'log'
        label_x = 'Período'
        label_y = 'Valores'
        eixo = True
        outliers = False
        size = (14,6)
        estados_br = False

        plot_box_plot(dados_boxplot,labels,nome_fig,escala,label_x,label_y,eixo,outliers,size,estados_br)

def plot_box_plot_estados(dfs,lat,lon,shp,var):

    # Cria um dataframe
    df = pd.DataFrame({
        'longitude': lon, # Adiciona uma coluna com longitude
        'latitude': lat # Adiciona uma coluna com latitude
    })
    
    # Cria a coluna geometria a partir de longitude e latitude no DataFrame
    df["geometry"] = df.apply(lambda row: Point(row["longitude"], row["latitude"]), axis=1)
    
    # Converte o df para GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

    gdf = gdf.to_crs("EPSG:31983")

    gdf['geometry'] = gdf.geometry.apply(lambda point: point_to_square(point,5750))

    gdf = gdf.to_crs("EPSG:4674")

    for freq in ['monthly', 'hourly']:
        
        for estado in shp["SIGLA_UF"]:
            print(estado)
            shp_estado = shp[shp["SIGLA_UF"]==estado]
            
            # Definição dos rótulos
            if freq == 'monthly':
                labels = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
                          'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']

            elif freq == 'weekly':
                labels = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sab', 'Dom']

            elif freq == 'hourly':
                labels = list(np.arange(24))
        
            dados_boxplot = []

            # Percorre cada DataFrame daquela frequência
            for i in dfs[var][freq]:
                
                df = dfs[var][freq][i]

                df['geometry'] = gdf.loc[df.index, "geometry"]

                df = gpd.GeoDataFrame(df,geometry="geometry",crs=gdf.crs)

                df = gpd.clip(df, shp_estado)

                df = df.drop(columns=['geometry'])

                #print(df)

                # Transforma em array 1D
                valores = df.values.flatten()

                #print('Valores são')
                #print(valores)

                # Filtra valores maiores que 0
                valores = valores[valores > 0]

                #print('Valores sem 0 são')
                #print(valores)

                dados_boxplot.append(valores)

            nome_fig = var+'_'+freq+'_'+estado
            escala = 'log'
            label_x = 'Período'
            label_y = 'Valores'
            eixo = True
            outliers = False
            size = (14,6)
            estados_br = False

            plot_box_plot(dados_boxplot,labels,nome_fig,escala,label_x,label_y,eixo,outliers,size,estados_br)

def ioapiCoords(ds):
    # Latlon
    lonI = ds.XORIG
    latI = ds.YORIG
    
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

def latlon_2d(dir_data):
   
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
    data = xr.open_dataset(dir_data)
    
    #processing coordinates
    xv, yv, lon, lat = ioapiCoords(data)
    xlon, ylat = eqmerc2latlon(data, xv, yv)
    lon2d = xlon.flatten()
    lat2d = ylat.flatten()
    
    return lon2d, lat2d

def point_to_square(point, tamanho):
    
    x,y = point.x,point.y
    
    return box(x-tamanho,y-tamanho,x+tamanho,y+tamanho)

def highTime(dfs, lat, lon, shp, var, ind_val):
    '''
    Identifica o período com a maior emissão para cada pixel (linha) e cada setor (coluna)
    e seus respectivos valores.
    
    Parameters:
    - dfs: list of DataFrames, each corresponding to a different time period (e.g., months, days, hours),
           where each column represents an emission sector.
    - time_labels: list of strings representing the labels for each time period, in the same order as the DataFrames.
    
    Returns:
    - A DataFrame where each cell contains the label of the time period with the highest emission for the respective pixel and sector.

    Parameters
    ----------
    dfs : Dictionary ou DataFrame 
            que possui os valores de emissão para cada poluente em cada pixel, 
            qual o maior emissor e qual a soma de emissão
    lat : Array of float64 (e.g. (21904,)) 
            com os valores de latitude para todos os pixels
    lon : Array of float64 (e.g. (21904,)) 
            com os valores de longitude para todos os pixels
        DESCRIPTION.
    shp : DataFrame
            com coluna geometry que possui um POLYGON
    freq : string
            que indica a frequência temporal (e.g. hourly, weekly, monthly, yearly)
    var : string
            que indica o poluente que está sendo avaliado
    ind_val : string
            indica se fará uma imagem dos índices (e.g. hora com o maior valor) 
            ou então valores (e.g. maior valor dentre as horas)

    Returns
    -------
    None

    '''
    
    for freq in ['monthly','hourly']:
        
        # Condição if para verificar a frequência e assim gerar o label para a plotagem das figuras
        if freq == 'monthly':
            labels = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']    
        elif freq == 'weekly':
            labels = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sab', 'Dom']
        elif freq == 'hourly':
            labels = list(np.arange(24))
        
        # Criar uma lista para armazenar a chave com maior valor para cada linha
        max_keys = []
        
        # Transforma os DataFrames em uma única matriz com as chaves
        df_combined = pd.DataFrame({key: df['wbd'] for key, df in dfs[var][freq].items()})
        
        # Pega a chave correspondente ao maior valor em cada linha
        max_keys = df_combined.idxmax(axis=1)  # Obtém os índices dos maiores valores
        max_keys[df_combined.nunique(axis=1) == 1] = np.nan  # Substitui por NaN se todos os valores forem iguais
        max_keys = max_keys.tolist()  # Converte para lista os índices com maior valor
        max_values = df_combined.max(axis=1).tolist() # Converte para lista os maiores valores
        
        # Cria um dataframe
        df = pd.DataFrame({
            'longitude': lon, # Adiciona uma coluna com longitude
            'latitude': lat, # Adiciona uma coluna com latitude
            'indice': max_keys, # Adiciona uma coluna com os índices que possuem maior valor
            'valor': max_values}) # Adiciona uma coluna com os maiores valores
        
        # Cria a coluna geometria a partir de longitude e latitude no DataFrame
        df["geometry"] = df.apply(lambda row: Point(row["longitude"], row["latitude"]), axis=1)
        
        # Converte o df para GeoDataFrame
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
        
        gdf = gdf.to_crs("EPSG:31983")
        
        gdf['geometry'] = gdf.geometry.apply(lambda point: point_to_square(point,5750))
        
        gdf = gdf.to_crs("EPSG:4674")
        
        # Cria o Envelope do Buffer
        #gdf['envelope'] = gdf.geometry.buffer(0.0135).envelope  
        
        # Cria um GeoDataFrame apenas com o envelope
        gdf = gpd.GeoDataFrame(gdf, geometry='geometry', crs="EPSG:4674")
        print(freq)
        for i in range(2): 
            
            # Cria um colormap discreto baseado na quantidade de categorias
            if freq == "monthly":

                cores = [
                         "darkred",      # 1
                         "red",          # 2
                         "orange",       # 3
                         "lightblue",    # 4
                         "paleturquoise", # 5
                         "lightcyan",    # 6
                         "dodgerblue",   # 7
                         "blue",         # 8
                         "navy",         # 9
                         "lightsalmon",  # 10
                         "peachpuff",    # 11
                         "lightpink",    # 12
                         ]

                cmap = ListedColormap(cores)

            else:

                cmap = plt.get_cmap('jet', len(labels))
            

            # Condição if para verificar se ind_val é indice ou valor
            if ind_val == 'indice':

                # Crie a normalização
                norm = mcolors.Normalize(vmin=0, vmax=len(labels))

                # Crie um mapeador de cores
                sm = cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])  # Necessário para que funcione com colorbar

            elif ind_val == 'valor':
                # Cria uma escala logarítmica para valor
                norm = mcolors.LogNorm(vmin=gdf['valor'].min()+10**-5, vmax=gdf['valor'].max())
            
            if i == 1:
                
                # Define o tamanho da figura
                fig, ax = plt.subplots(figsize=(12, 12))
                
                # Plota gdf
                gdf.plot(column=ind_val, cmap=cmap, ax=ax, legend=False, norm = norm, alpha=1)
                
                # Plota o shapefile 'shp' com borda preta e face preta
                shp.plot(ax=ax, edgecolor='black', facecolor='none', alpha=0.5)
                
                # Adicionando o contexto de mapa do contextily (camada de fundo)
                #cx.add_basemap(
                #    ax,
                #    source=cx.providers.Esri.WorldImagery,
                #    crs=gdf.crs,
                #    alpha=0.5
                #)
                
                # Definir os limites do gráfico.....
                ax.set_xlim(-74, -35)
                ax.set_ylim(-34, 5)
                
                # Adicione o colorbar abaixo do gráfico com tamanho reduzido
                cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.1, shrink=0.85,alpha=1)  # shrink ajusta o tamanho
                
                # Adicione os rótulos personalizados
                cbar.set_ticks(range(len(labels)))  
                cbar.set_ticklabels(labels)
                
                cbar.ax.set_title({'monthly':'Mês', 'weekly':'Dia da Semana','hourly':'Hora do Dia'}.get(freq))
                
                # Salva a figura
                fig.savefig('/home/lcqar/BRAIN/emis/windBlowDustBR/Outputs/BR_12km/objetivo02/high_' + ind_val + '_' + var + '_' + freq + '.png')
                
            else:

                for estado in shp["SIGLA_UF"]:
                    print(estado)
                    # Define o tamanho da figura
                    fig, ax = plt.subplots(figsize=(12, 12))

                    # Cria um colormap discreto baseado na quantidade de categorias
                    if freq == "monthly":
                        cores = [
                                 "darkred",       # 1
                                 "red",           # 2
                                 "orange",        # 3
                                 "lightblue",     # 4
                                 "paleturquoise", # 5
                                 "lightcyan",     # 6
                                 "dodgerblue",    # 7
                                 "blue",          # 8
                                 "navy",          # 9
                                 "lightsalmon",   # 10
                                 "peachpuff",     # 11
                                 "lightpink",     # 12
                                 ]
                        cmap = ListedColormap(cores)

                    else:
                        cmap = plt.get_cmap('jet', len(labels))

                    shp_estado = shp[shp["SIGLA_UF"]==estado]
                    minx,miny,maxx,maxy = shp_estado.total_bounds

                    gdf_estado = gpd.clip(gdf, shp_estado)

                    # Plota gdf
                    gdf_estado.plot(column=ind_val, cmap=cmap, ax=ax, legend=False, norm = norm, alpha=1)

                    # Plota o shapefile 'shp' com borda preta e face preta
                    shp.plot(ax=ax, edgecolor='black', facecolor='none', alpha=0.5)

                    # Adicionando o contexto de mapa do contextily (camada de fundo)
                    #cx.add_basemap(
                    #    ax,
                    #    source=cx.providers.Esri.WorldImagery,
                    #    crs=gdf.crs,
                    #    alpha=0.5
                    #)
                    
                    # Definir os limites do gráfico.....
                    ax.set_xlim(minx, maxx)
                    ax.set_ylim(miny, maxy)
                    
                    # Adicione o colorbar abaixo do gráfico com tamanho reduzido
                    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.1, shrink=0.95, alpha=1)  # shrink ajusta o tamanho
                    
                    # Adicione os rótulos personalizados
                    cbar.set_ticks(range(len(labels)))  
                    cbar.set_ticklabels(labels)
                    
                    cbar.ax.set_title({'monthly':'Mês', 'weekly':'Dia da Semana','hourly':'Hora do Dia'}.get(freq))
                    
                    # Salva a figura
                    fig.savefig('/home/lcqar/BRAIN/emis/windBlowDustBR/Outputs/BR_12km/objetivo02/high_' + ind_val + '_' + var + '_' + freq + '_' + estado + '.png')
                    
        # Define a pasta de destino para salvar o gdf
        #output_folder = r'/home/artaxo/CMAQ_REPOv5.4/jose/geodfs'  # Substitua pelo seu caminho real
        
        # Caminho do arquivo CSV
        #output_path = os.path.join(output_folder, f"high_{ind_val}_{var}_{freq}.csv")
        
        # Converte a geometria para WKT e salva em csv
        #gdf["geometry"] = gdf["geometry"].apply(lambda geom: geom.wkt)  # Converte para texto
        #gdf.drop(columns=["geometry"], inplace=True)  # Remove as colunas geometry e envelope
        #gdf.to_csv(output_path, index=False) 

##################################################################################################

'''
final_dict = agg_wbd(path,np.sum)

for var in ['PM10','PMC','PMFINE']: 

    final_dict[var]['monthly'] = dict(sorted(final_dict[var]['monthly'].items()))
    final_dict[var]['monthly'] = {k - 1: v for k, v in final_dict[var]['monthly'].items()}

print(final_dict['PM10']['monthly'])

final_dict = agg_temporal(final_dict,np.sum)

with open(r"/home/lcqar/BRAIN/emis/windBlowDustBR/Outputs/BR_12km/objetivo02/emissoes_wbd.pkl", "wb") as f:
    pickle.dump(final_dict, f)

'''

with open(r"/home/lcqar/BRAIN/emis/windBlowDustBR/Outputs/BR_12km/objetivo02/emissoes_wbd.pkl", 'rb') as f:
    dict_dfs = pickle.load(f)

lon, lat = latlon_2d("/home/lcqar/BRAIN/emis/windBlowDustBR/Outputs/BR_12km/windBlowDust_PM10_2023-02-24-00:00:00_2023-02-25-00:00:00.nc")
shp = gpd.read_file("/home/lcqar/BRAIN/emis/windBlowDustBR/inputs/shapefiles/BR_UF_2024.shp",engine="pyogrio")

#for var in ['PM10','PMC','PMFINE']:
    #highTime(dict_dfs, lat, lon, shp, var, 'indice')
    #plot_box_plot_ano_estado(dict_dfs,lat,lon,shp,var)
    #plot_box_plot_estados(dict_dfs, lat, lon, shp, var)
    #plot_box_plot_brasil(dict_dfs,shp,var)
    #plot_box_plot_ano_solo(dict_dfs, lat, lon, shp, var)

import xarray as xr
import numpy as np
import glob

base = "/home/lcqar/BRAIN/emis/windBlowDustBR/Outputs/BR_12km"

arquivos = []

# pega todos os arquivos windBlowDust_PM10 em todas as pastas
for pasta in sorted(glob.glob(base)):
    arquivos.extend(glob.glob(pasta + "/windBlowDust_PM10*"))

# abre todos e concatena no TSTEP
ds = xr.open_mfdataset(arquivos, concat_dim="TSTEP", combine="nested")

# remove dimensão LAY se for 1
pm10 = ds['PM10']

# transforma em array numpy
dados = pm10.values

# pega os 10 maiores valores
indices = np.argpartition(dados.flatten(), -10)[-10:]
indices = indices[np.argsort(dados.flatten()[indices])[::-1]]

# converte para indices TSTEP, ROW, COL
t, r, c = np.unravel_index(indices, dados.shape)

valores = dados[t, r, c]

for i in range(10):
    print(f"{i+1}: Valor={valores[i]:.2f} | TSTEP={t[i]} ROW={r[i]} COL={c[i]}")
