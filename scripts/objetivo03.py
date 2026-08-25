import netCDF4 as nc
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pathlib import Path
import os
import glob
import shutil
import tempfile
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pyproj
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import calendar
import pickle
import pyproj
import xarray as xr
import regridMAPBIOMAS as regMap
import gridDetails as grd
from shapely.geometry import Point, box
from matplotlib.colors import ListedColormap, LogNorm
from scipy.stats import kruskal
from matplotlib.patches import Patch

#%% Abrindo arquivos

ds_wbd_mp10 = xr.open_dataset('/home/lcqar/UTILS/outputs/windblowdust_only/BRAIN_BASECONC_PM10_2023_09_01_00_to_2023_10_01_23.nc'
                              )['PM10'][-528:-24,0,:,:]

ds_all_mp10 = xr.open_dataset('/home/lcqar/UTILS/outputs/all_emis/BRAIN_BASECONC_PM10_2023_08_26_00_to_2023_10_01_23.nc'
                              )['PM10'][-528:-24,0,:,:]

serie_wbd = ds_wbd_mp10.max(dim=('ROW', 'COL'))
mean_wbd = ds_wbd_mp10.mean(dim=('ROW', 'COL'))
serie_all = ds_all_mp10.mean(dim=('ROW', 'COL'))

fig, ax = plt.subplots(figsize=(12,4))

ax.plot(serie_wbd['TSTEP'], serie_wbd, label='max WBD')
ax.plot(mean_wbd['TSTEP'], mean_wbd, label='mean WBD')
ax.plot(serie_all['TSTEP'], serie_all, label='ALL')

ax.set_xlabel('TSTEP')
ax.set_ylabel('PM10 médio')
ax.legend()
ax.grid(True)

plt.tight_layout()

fig.savefig(
    'serie_temporal_pm10_medio.png',
    dpi=300,
    bbox_inches='tight'
)

plt.close(fig)

# Máscara dos casos onde WBD > ALL
mask = ds_wbd_mp10.values > ds_all_mp10.values

# Índices dos casos verdadeiros
tstep, row, col = np.where(mask)

# Valores correspondentes
wbd_vals = ds_wbd_mp10.values[tstep, row, col]
all_vals = ds_all_mp10.values[tstep, row, col]

# DataFrame para inspeção
df_sup = pd.DataFrame({
    'TSTEP': tstep,
    'ROW': row,
    'COL': col,
    'WBD': wbd_vals,
    'ALL': all_vals,
    'DIF': wbd_vals - all_vals
})

print(f'{len(df_sup)} ocorrências encontradas')

# opcional: ordenar pela maior diferença
df_sup = df_sup.sort_values('DIF', ascending=False)

print(df_sup.head(20))

def pegar_primeiros_24(ds):
    return ds.isel(TSTEP=slice(0, 24))

ds_emis_mp10 = xr.open_mfdataset(
    "/home/lcqar/BRAIN/emis/windBlowDustBR/Outputs/BR_12km/windBlowDust_PM10_2023-09*.nc",
    preprocess=pegar_primeiros_24,
    combine='nested',       # Empilha os arquivos um atrás do outro
    concat_dim='TSTEP'      # Diz explicitamente qual dimensão deve ser expandida
    )['PM10'][-504:,:,:]*3600*1e-6

ds_mask = ds_emis_mp10.sum(dim='TSTEP')

ds_wbd_mp10_int = ds_wbd_mp10
ds_all_mp10_int = ds_all_mp10

ds_wbd_mp10 = ds_wbd_mp10.where(ds_mask != 0, 0)
ds_all_mp10 = ds_all_mp10.where(ds_mask != 0, 0)

ds_wbd_10_dia = ds_wbd_mp10.coarsen(TSTEP=24, boundary='trim').mean()
ds_all_10_dia = ds_all_mp10.coarsen(TSTEP=24, boundary='trim').mean()
ds_emis_10_dia = ds_emis_mp10.coarsen(TSTEP=24 ).sum()

ds_wbd_10_mes = ds_wbd_mp10.mean(dim='TSTEP')
ds_all_10_mes = ds_all_mp10.mean(dim='TSTEP')
ds_emis_10_mes = ds_mask

print(ds_wbd_mp10.shape)
print(ds_all_mp10.shape)
print(ds_emis_mp10.shape)

PI_MP10 = 10
PI_MP25 = 10
PF_MP10 = 10
PF_MP25 = 10

SHP_PATH   = "/home/lcqar/BRAIN/emis/windBlowDustBR/inputs/shapefiles/BR_UF_2024.shp"
OUTPUT_DIR = "/home/lcqar/BRAIN/emis/windBlowDustBR/Outputs/BR_12km/objetivo03/"

REGIOES = {
    'South':          {'cor': '#e74c3c', 'estados': ['RS', 'SC', 'PR']},           # vermelho
    'Southeast':      {'cor': '#3498db', 'estados': ['SP', 'MG', 'RJ', 'ES']},     # azul
    'Midwest':        {'cor': '#9b59b6', 'estados': ['MT', 'MS', 'GO', 'DF']},     # roxo
    'Northeast':      {'cor': '#f1c40f', 'estados': ['BA', 'SE', 'AL', 'PE', 'PB', 'RN', 'CE', 'PI', 'MA']},  # amarelo
    'North':          {'cor': '#2ecc71', 'estados': ['TO', 'PA', 'AP', 'RR', 'AM', 'AC', 'RO']},              # verde
}

# Ordem padronizada de estados por região (Sul → Norte geográfico)
ESTADOS_ORDEM = ['RS','SC','PR','SP','MG','RJ','ES',
                 'BA','SE','AL','PE','PB','RN','CE','PI','MA',
                 'TO','PA','AP','RR','AM','AC','RO',
                 'MT','MS','GO','DF']

CORES_MENSAIS = [
    "darkred","red","orange","lightblue","paleturquoise","lightcyan",
    "dodgerblue","blue","navy","lightsalmon","peachpuff","lightpink"
]

LABELS_MENSAIS  = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

LABELS_HORARIOS = [str(h) for h in range(24)]

# Solos especiais (requerem máscara MapBiomas – substitua pelas suas máscaras)
SOLOS_ESPECIAIS = {'Solo Duna': None, 'Solo Mineração': None}

# ── Usos do solo e suas cores ─────────────────────────────────────────────────
USOS_SOLO = {
    'Dune / Sandy area': '#D2A679',   # caramelo
    'Mining':            '#808080',   # cinza
}

# Cores das regiões (para barras empilhadas)
CORES_REGIOES = {r: REGIOES[r]['cor'] for r in REGIOES}

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

def _point_to_square(point, tamanho=8000):
    """Converte ponto projetado em quadrado de lado 2*tamanho."""
    x, y = point.x, point.y
    return box(x - tamanho, y - tamanho, x + tamanho, y + tamanho)


def _build_gdf(lat, lon):
    """Cria GeoDataFrame de pixels com geometria quadrada (EPSG:4674)."""
    df = pd.DataFrame({'longitude': lon, 'latitude': lat})
    df['geometry'] = df.apply(lambda r: Point(r['longitude'], r['latitude']), axis=1)
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
    gdf = gdf.to_crs('EPSG:31983')
    gdf['geometry'] = gdf.geometry.apply(_point_to_square)
    gdf = gdf.to_crs('EPSG:4674')
    return gdf


def _mask_regiao(gdf_pixels, shp, escala_espacial):
    """
    Retorna (gdf_recortado, lista_de_estados_presentes | None).

    Para 'Brasil', 'Solo Duna', 'Solo Mineração' retorna o gdf completo
    e None como lista de estados.
    """
    if escala_espacial == 'Brasil':
        return gdf_pixels.copy(), None

    if escala_espacial in SOLOS_ESPECIAIS:
        mascara = SOLOS_ESPECIAIS[escala_espacial]
        if mascara is not None:
            return gpd.clip(gdf_pixels, mascara), None
        # Sem máscara definida – retorna tudo com aviso
        print(f"[AVISO] Máscara para '{escala_espacial}' não definida. Usando Brasil inteiro.")
        return gdf_pixels.copy(), None

    # Regiões ou estados
    if escala_espacial in REGIOES:
        estados = REGIOES[escala_espacial]['estados']
    else:
        # Assume que é um estado individual
        estados = [escala_espacial]

    shp_sel = shp[shp['SIGLA_UF'].isin(estados)]
    gdf_rec = gpd.clip(gdf_pixels, shp_sel)
    return gdf_rec, estados


# ──────────────────────────────────────────────────────────────────────────────
# Cálculos pixel a pixel
# ──────────────────────────────────────────────────────────────────────────────

def _pixel_dataframe(dfs, var, freq):
    """
    Retorna DataFrame (n_pixels × n_periodos) com as emissões somadas
    para cada pixel em cada período temporal.

    A chave de cada período é 0-based (0..11 para mensal, 0..23 para horária).
    """
    periodos = sorted(dfs[var][freq].keys())
    cols = {}
    for t in periodos:
        df_t = dfs[var][freq][t]
        # Pode ter múltiplas colunas (anos/meses acumulados) → soma por linha
        cols[t] = df_t.sum(axis=1).values
    return pd.DataFrame(cols)           # shape: (n_pixels, n_periodos)


def _calc_metrica(pixel_df, tipo, freq):
    """
    Calcula a métrica escolhida para cada pixel.

    Retorna Series de tamanho n_pixels.

    tipo: 'pixel_max' | 'msi' | 'amplitude' | 'forca'
    """
    vals = pixel_df.values          # (n_pixels, n_periodos)
    n_periodos = vals.shape[1]

    if tipo == 'pixel_max':
        # Índice do período com maior emissão (0-based)
        result = np.argmax(vals, axis=1).astype(float)
        # Pixels com todos os valores iguais (ex: zero) → NaN
        result[np.all(vals == vals[:, [0]], axis=1)] = np.nan
        return pd.Series(result, index=pixel_df.index)

    elif tipo == 'amplitude':
        monthly_means = vals                        # já é soma por período
        amp = np.nanmax(monthly_means, axis=1) - np.nanmin(monthly_means, axis=1)
        return pd.Series(amp, index=pixel_df.index)

    elif tipo == 'forca':
        var_total    = np.nanvar(vals, axis=1)
        monthly_mean = np.nanmean(vals, axis=1, keepdims=True)  # média temporal por pixel
        var_sazonal  = np.nanvar(vals - monthly_mean, axis=1)   # variância entre períodos
        # Força relativa = variância entre médias dos períodos / variância total
        periodos_means = np.nanmean(vals, axis=0, keepdims=True) # (1, n_periodos)
        var_periodos = np.nanvar(
            np.tile(periodos_means, (vals.shape[0], 1)) - np.nanmean(vals),
            axis=1
        )
        # Implementação fiel ao objetivo02.py: var(médias_mensais) / var_total
        medias_por_periodo = np.nanmean(vals, axis=0)           # (n_periodos,)
        var_medias  = np.nanvar(medias_por_periodo)             # escalar global
        forca = np.where(var_total > 0, var_medias / var_total, np.nan)
        return pd.Series(forca, index=pixel_df.index)

    elif tipo == 'msi':
        # Markham Seasonality Index por pixel
        total = np.nansum(vals, axis=1, keepdims=True)
        total = np.where(total == 0, np.nan, total)
        rel_freq = vals / total                                 # (n_pixels, n_periodos)
        msi = np.nansum(np.abs(rel_freq - 1/n_periodos), axis=1) / (2 * (1 - 1/n_periodos))
        return pd.Series(msi, index=pixel_df.index)

    else:
        raise ValueError(f"Tipo desconhecido: '{tipo}'. Use 'pixel_max','msi','amplitude','forca'.")

def _uso_solo(pixel_df, var, escala_espacial):
    """
    Retorna pixel_df ponderado pela fração de uso do solo do tipo escolhido.

    O peso `content` varia de 0 a 1 e representa a fração da área de cada
    pixel ocupada por duna ou mineração, calculada a partir do MapBiomas via
    regMap / grd. O pixel_df resultante mantém a mesma forma e índice do
    original — apenas os valores são multiplicados pelo peso.

    Parâmetros
    ----------
    pixel_df : DataFrame  (n_pixels × n_periodos)
        Emissões brutas de cada pixel em cada período temporal.
    var : str
        Poluente ('PM10', 'PMC', 'PMFINE') — não usado no cálculo do peso,
        mantido por consistência com a assinatura da chamada.
    escala_espacial : str
        'Solo Duna' ou 'Solo Mineração'.

    Retorno
    -------   pixel_df_pond : DataFrame  (n_pixels × n_periodos)
        Emissões ponderadas pela fração de uso do solo.
    content : np.ndarray  (n_pixels,)
        Vetor de pesos [0, 1] para uso posterior (mapa, boxplot).
    """
    import gridDetails as grd
    import regridMAPBIOMAS as regMap

    # ── Leitura das grades ────────────────────────────────────────────────────
    (ds, datesTime, lia, domainShp,
     lat, lon, lat_index, lon_index, grids) = grd.main(
        '/home/lcqar/GAR_BR/mcip/BR_12km/METCRO3D_BR_12km_2023-09-27.nc',
        '/home/lcqar/GAR_BR/mcip/BR_12km/GRIDDOT2D_BR_12km_2023-09-27.nc',
        '/home/lcqar/GAR_BR/WRF/2023/2023_09', 'd02'
    )

    av, al, alarea, lat, lon, domainShp = regMap.main(
        'BR_12km', 'inputFolder',
        '/home/lcqar/BRAIN/emis/windBlowDustBR/Outputs/BR_12km',
        2023, [23, 30], False,
        grids, domainShp, lat, lon
    )

    # ── Frações de uso do solo ────────────────────────────────────────────────
    # alarea[0] = duna, alarea[1] = mineração  (área em cada pixel)
    area_duna  = alarea[0, :, :].flatten()
    area_mine  = alarea[1, :, :].flatten()
    area_total = area_duna + area_mine

    # Evita divisão por zero (pixels sem nenhum dos dois usos → peso = 0)
    with np.errstate(invalid='ignore', divide='ignore'):
        content_dune = np.where(area_total > 0, area_duna / area_total, 0.0)
        content_mine = np.where(area_total > 0, area_mine / area_total, 0.0)

    print(f"[_uso_solo] content_dune  max={content_dune.max():.4f}  "
          f"pixels>0: {(content_dune > 0).sum()}")
    print(f"[_uso_solo] content_mine  max={content_mine.max():.4f}  "
          f"pixels>0: {(content_mine > 0).sum()}")

    # ── Seleciona o peso correto ──────────────────────────────────────────────
    if escala_espacial == 'Solo Duna':
        content = content_dune
    elif escala_espacial == 'Solo Mineração':
        content = content_mine
    else:
        raise ValueError(
            f"_uso_solo chamada com escala_espacial='{escala_espacial}'. "
            "Use 'Solo Duna' ou 'Solo Mineração'."
        )

    # ── Pondera pixel_df ──────────────────────────────────────────────────────
    # content tem n_pixels entradas; pixel_df tem n_pixels linhas (mesmo índice)
    content_series = pd.Series(content, index=pixel_df.index)
    pixel_df_pond  = pixel_df.multiply(content_series, axis=0)

    return pixel_df_pond, content

# ──────────────────────────────────────────────────────────────────────────────
# Plot do mapa (painel esquerdo)
# ──────────────────────────────────────────────────────────────────────────────

nc_exemplo = (
    "/home/lcqar/BRAIN/emis/windBlowDustBR/Outputs/BR_12km/"
    "windBlowDust_PM10_2023-02-24-00:00:00_2023-02-25-00:00:00.nc"
)

lon, lat = latlon_2d(nc_exemplo)
shp = gpd.read_file(SHP_PATH, engine="pyogrio")

regioes = {
    'RS': 'Sul',
    'SC': 'Sul',
    'PR': 'Sul',

    'SP': 'Sudeste',
    'RJ': 'Sudeste',
    'MG': 'Sudeste',
    'ES': 'Sudeste',

    'MS': 'Centro-Oeste',
    'MT': 'Centro-Oeste',
    'GO': 'Centro-Oeste',
    'DF': 'Centro-Oeste',

    'BA': 'Nordeste',
    'SE': 'Nordeste',
    'AL': 'Nordeste',
    'PE': 'Nordeste',
    'PB': 'Nordeste',
    'RN': 'Nordeste',
    'CE': 'Nordeste',
    'PI': 'Nordeste',
    'MA': 'Nordeste',

    'AC': 'Norte',
    'AM': 'Norte',
    'RR': 'Norte',
    'RO': 'Norte',
    'PA': 'Norte',
    'AP': 'Norte',
    'TO': 'Norte'
}

shp2=shp
shp2['REGIAO']=shp2['SIGLA_UF'].map(regioes)
shp2=shp2.dissolve(by='REGIAO').reset_index()

#%%

# =============================================================================
# FIGURA 1: SCATTER PLOT E MAPA (QUADRANTES DE IMPACTO)
# =============================================================================

def plot_impacto_quadrantes(
    ds_all_10_dia, ds_emis_10_dia, lat, lon, shp, 
    PI_MP10=100, salvar=True
):
    """
    Cria a Figura 1:
    - Esquerda: Mapa com a cor mais 'crítica' que o pixel atingiu ao longo dos dias.
    - Direita: Scatter plot (Emissão x Concentração) dividido em 4 quadrantes.
    """
    print("[INFO] Gerando Figura 1 - Scatter e Mapa de Quadrantes...")
    
    # Extrai os valores brutos para numpy arrays (achatados para cálculo da mediana)
    emis_vals = ds_emis_10_dia.values
    conc_vals = ds_all_10_dia.values

    # 1. Calcula a mediana das emissões APENAS onde existe emissão (valores > 0)
    if len(emis_vals) == 0:
        print("[AVISO] Nenhuma emissão > 0 encontrada.")
        mediana_emis = 0
    else:
        mediana_emis = np.nanmedian(emis_vals[emis_vals>0])
        
    print(f"[INFO] Mediana da emissão (x-axis): {mediana_emis:.4f}")

    # 2. Classificação de cada pixel no tempo e no espaço (1 a 4)
    # 1: Azul (Y <= PI, X <= Mediana)
    # 2: Verde (Y <= PI, X > Mediana)
    # 3: Laranja (Y > PI, X <= Mediana)
    # 4: Vermelho (Y > PI, X > Mediana)
    # 0: NaN ou Mascarado
    
    classificacao = xr.zeros_like(ds_all_10_dia, dtype=int)
    
    # Aplicando as condições:
    # Máscara de pixels válidos (onde ds_all_10_dia não é nulo/zero se foi mascarado)
    mascara_validos = (ds_all_10_dia > 0) | (ds_emis_10_dia > 0)
    
    classificacao = xr.where(mascara_validos & (ds_all_10_dia <= PI_MP10) & (ds_emis_10_dia <= mediana_emis), 1, classificacao)
    classificacao = xr.where(mascara_validos & (ds_all_10_dia <= PI_MP10) & (ds_emis_10_dia >  mediana_emis), 2, classificacao)
    classificacao = xr.where(mascara_validos & (ds_all_10_dia >  PI_MP10) & (ds_emis_10_dia <= mediana_emis), 3, classificacao)
    classificacao = xr.where(mascara_validos & (ds_all_10_dia >  PI_MP10) & (ds_emis_10_dia >  mediana_emis), 4, classificacao)

    # 3. Mapa: Colapsa o tempo pegando a PIOR classificação do pixel no mês (o máximo resolve isso perfeitamente!)
    mapa_cores = classificacao.max(dim='TSTEP')
    
    # Monta o GeoDataFrame para plotar no mesmo formato
    gdf_pixels = _build_gdf(lat.flatten(), lon.flatten()) # Função do seu arquivo
    gdf_plot = gdf_pixels.copy()
    gdf_plot['metrica'] = mapa_cores.values.flatten()
    
    # Remove pixels zerados (mascarados)
    gdf_plot = gdf_plot[gdf_plot['metrica'] > 0]

    # 4. Criando a Figura
    fig, (ax_map, ax_scatter) = plt.subplots(
        1, 2, figsize=(14, 6),
        gridspec_kw={'width_ratios': [1.2, 1], 'wspace': 0.15}
    )

    # --- PLOT MAPA ---
    cores_quadrantes = ['#3498db', '#2ecc71', '#e67e22', '#e74c3c'] # Azul, Verde, Laranja, Vermelho
    cmap = ListedColormap(cores_quadrantes)
    norm = mcolors.BoundaryNorm(boundaries=[0.5, 1.5, 2.5, 3.5, 4.5], ncolors=4)

    shp.plot(ax=ax_map, edgecolor='white', color='gainsboro', linewidth=0.5, alpha=0.7)
    gdf_plot.plot(column='metrica', cmap=cmap, norm=norm, ax=ax_map, legend=False, alpha=1)
    
    ax_map.set_xlim(-74, -34); ax_map.set_ylim(-34, 6)
    ax_map.set_aspect('equal')
    ax_map.set_axis_off()

    # --- PLOT SCATTER ---
    # Preparando dados 1D validos para o scatter plot
    mask_1d = mascara_validos.values.flatten()
    x_scat = emis_vals.flatten()[mask_1d]
    y_scat = conc_vals.flatten()[mask_1d]
    c_scat = classificacao.values.flatten()[mask_1d]
    
    # Mapeando os inteiros 1-4 para as cores
    cor_map_array = np.array(['white', '#3498db', '#2ecc71', '#e67e22', '#e74c3c'])
    cores_scatter = cor_map_array[c_scat]

    # Plotando os pontos
    ax_scatter.scatter(x_scat, y_scat, c=cores_scatter, s=8, alpha=0.6, edgecolors='none')
    ax_scatter.set_xscale('log')
    ax_scatter.set_ylim(bottom=0)

    # Linhas divisórias (Quadrantes)
    ax_scatter.axhline(PI_MP10, color='gray', linestyle='--', linewidth=1.5, label=f'PI MP10 ({PI_MP10})')
    ax_scatter.axvline(mediana_emis, color='gray', linestyle=':', linewidth=1.5, label=f'Mediana Emissão ({mediana_emis:.2f})')
    
    ax_scatter.set_xlabel('Emissão Dust In the Wind de PM10 (ton)', fontsize=11)
    ax_scatter.set_ylabel(r'Concentração Total PM10 ($\mu g/m^3$)', fontsize=11)
    ax_scatter.set_title('Relação Concentração x Emissão', fontsize=12)
    ax_scatter.grid(True, alpha=0.3)
    ax_scatter.legend()
    
    plt.tight_layout()
    if salvar:
        fig.savefig(OUTPUT_DIR + r"figura1_quadrantes_impacto.png", dpi=150, bbox_inches='tight')
    plt.show()

# =============================================================================
# FIGURA 2: MAPA DE RAZÃO E BOXPLOT POR REGIÕES
# =============================================================================

def _uso_solo(razao_vals, var):

    """
    Retorna pixel_df ponderado pela fração de uso do solo do tipo escolhido.
    O peso `content` varia de 0 a 1 e representa a fração da área de cada
    pixel ocupada por duna ou mineração, calculada a partir do MapBiomas via
    regMap / grd. O pixel_df resultante mantém a mesma forma e índice do
    original — apenas os valores são multiplicados pelo peso.

    Parâmetros
    ----------
    pixel_df : DataFrame  (n_pixels × n_periodos)
        Emissões brutas de cada pixel em cada período temporal.
    var : str
        Poluente ('PM10', 'PMC', 'PMFINE') — não usado no cálculo do peso,
        mantido por consistência com a assinatura da chamada.
    escala_espacial : str
        'Solo Duna' ou 'Solo Mineração'.

    Retorno
    -------
    pixel_df_pond : DataFrame  (n_pixels × n_periodos)
        Emissões ponderadas pela fração de uso do solo.
    content : np.ndarray  (n_pixels,)
        Vetor de pesos [0, 1] para uso posterior (mapa, boxplot).
    """
    import gridDetails as grd
    import regridMAPBIOMAS as regMap

    # ── Leitura das grades ────────────────────────────────────────────────────

    (ds, datesTime,lia,domainShp,
     lat, lon, lat_index, lon_index, grids) = grd.main(
        '/home/lcqar/GAR_BR/mcip/BR_12km/METCRO3D_BR_12km_2023-09-27.nc',
        '/home/lcqar/GAR_BR/mcip/BR_12km/GRIDDOT2D_BR_12km_2023-09-27.nc',
        '/home/lcqar/GAR_BR/WRF/2023/2023_09', 'd02'
    )

    av, al, alarea, lat, lon, domainShp = regMap.main(
        'BR_12km', 'inputFolder',
        "/home/lcqar/BRAIN/emis/windBlowDustBR/Outputs/BR_12km",
        2023, [23, 30], False,
        grids, domainShp, lat, lon
    )

    # ── Frações de uso do solo ────────────────────────────────────────────────
    # alarea[0] = duna, alarea[1] = mineração  (área em cada pixel)
    area_dune  = alarea[0, :, :]
    area_mine  = alarea[1, :, :]

    area_mine[area_mine>0] = 1
    area_dune[area_dune>0] = 1

    razao_wbd_mine = area_mine*razao_vals
    razao_wbd_dune = area_dune*razao_vals

    return razao_wbd_mine, razao_wbd_dune

def estatisticas_boxplot(x):

    x = np.asarray(x)
    x = x[~np.isnan(x)]

    if len(x) == 0:
        return [np.nan]*8

    q1 = np.percentile(x, 25)
    med = np.percentile(x, 50)
    q3 = np.percentile(x, 75)

    iqr = q3 - q1

    whisker_inf = x[x >= q1 - 1.5*iqr].min()
    whisker_sup = x[x <= q3 + 1.5*iqr].max()

    return [
        len(x),          # Quantidade
        x.max(),         # Máximo
        whisker_sup,     # Whisker superior
        q3,              # Percentil 75
        med,             # Mediana
        q1,              # Percentil 25
        whisker_inf,     # Whisker inferior
        x.min()          # Mínimo
    ]

def plot_razao_wbd_solo(ds_wbd_10_mes, ds_all_10_mes, lat, lon, shp, salvar=True):
    """
    Cria a Figura 2:
    - Esquerda: Mapa com a razão (0 a 100%) entre PM10_WBD e PM10_ALL.
    - Direita: Boxplot horizontal pareado por Região e Uso do Solo (Dunas x Mineração).
    """
    print("[INFO] Gerando Figura 2 - Razão Percentual WBD/ALL por Uso do Solo...")

    # 1. Calcula a razão (%) 2D (Tratando divisão por zero e NaNs)
    razao = (ds_wbd_10_mes / ds_all_10_mes) * 100
    razao = xr.where(~np.isfinite(razao), 0, razao)

    # 2. Usa a sua função para aplicar as máscaras 2D
    # Passamos os valores 2D da razão para multiplicar pela máscara 2D do MapBiomas
    razao_wbd_mine, razao_wbd_dune = _uso_solo(razao, 'PM10')

    # 3. Cria GeoDataFrame com todas as informações "achatadas" (1D)
    gdf_pixels = _build_gdf(lat.flatten(), lon.flatten())
    gdf_plot = gdf_pixels.copy()
    gdf_plot['razao'] = razao.values.flatten()
    gdf_plot['razao_mine'] = razao_wbd_mine.flatten()
    gdf_plot['razao_dune'] = razao_wbd_dune.flatten()

    # 4. Criando a Figura
    fig, (ax_map, ax_box) = plt.subplots(
        1, 2, figsize=(14, 6),
        gridspec_kw={'width_ratios': [1.5, 0.4], 'wspace': -0.25}
    )

    # --- PLOT MAPA ---
    # Filtra apenas os pixels com alguma contribuição para o mapa
    gdf_mapa = gdf_plot[gdf_plot['razao'] > 0]

    cmap = plt.get_cmap('tab20b')
    norm = mcolors.Normalize(vmin=0, vmax=100)

    shp2.plot(ax=ax_map, edgecolor='white', color='gainsboro', linewidth=0.5, alpha=0.7)
    gdf_mapa.plot(column='razao', cmap=cmap, norm=norm, ax=ax_map, legend=False, alpha=1)

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = inset_axes(ax_map, width="3%", height="36%", loc='lower left',
                     bbox_to_anchor=(0.15, 0.05, 1, 1), bbox_transform=ax_map.transAxes, borderpad=0)
    cbar = plt.colorbar(sm, cax=cax, orientation='vertical')
    cbar.set_label('Contribution of DIW (%)', fontsize=9)
    cbar.ax.tick_params(labelsize=7)
    cbar.ax.yaxis.set_label_position('left')

    ax_map.set_xlim(-74, -34); ax_map.set_ylim(-34, 6)
    ax_map.set_aspect('equal')
    ax_map.set_axis_off()
    ax_map.set_title('(a)', fontsize=12, loc='left')

    # --- PLOT BOXPLOT PAREADO ---
    regioes_ordem = ['North','Northeast','Midwest','Southeast','South']
    siglas_map = {'South': 'S', 'Southeast': 'SE', 'Midwest': 'CO', 'Northeast': 'NE', 'North': 'N'}

    dados_duna = []
    dados_mine = []
    y_positions = np.arange(len(regioes_ordem))

    for regiao in regioes_ordem:
        estados = REGIOES[regiao]['estados']
        shp_reg = shp[shp['SIGLA_UF'].isin(estados)]

        # Recorta os pixels para a região atual
        gdf_reg = gpd.sjoin(
            gdf_plot,
            shp_reg[['geometry']],
            predicate='intersects',
            how='inner'
        ).drop(columns='index_right')

        # Coleta os valores usando a sua regra de > 0.1% para evitar excesso de zeros
        vals_duna = gdf_reg['razao_dune'].dropna().values
        vals_duna = vals_duna[vals_duna>0]

        vals_mine = gdf_reg['razao_mine'].dropna().values
        vals_mine = vals_mine[vals_mine>0]

        dados_duna.append(vals_duna if len(vals_duna) > 0 else np.array([np.nan]))
        dados_mine.append(vals_mine if len(vals_mine) > 0 else np.array([np.nan]))

    # Configurações visuais idênticas ao objetivo_02.py
    box_width = 0.28
    offset = 0.17
    cor_duna = '#D2A679'
    cor_mine = '#808080'

    flierprops = dict(marker='o', markersize=2, linestyle='none')

    # Boxplot Mineração (Deslocado para cima)
    bp_mine = ax_box.boxplot(
        dados_mine,
        positions=y_positions + offset,
        flierprops=flierprops,
        vert=False, widths=box_width, patch_artist=True, showfliers=True,
        boxprops=dict(facecolor='white', color=cor_mine, linewidth=1.5),
        whiskerprops=dict(color=cor_mine, linewidth=1.5),
        capprops=dict(color=cor_mine, linewidth=1.5),
        medianprops=dict(color=cor_mine, linewidth=1.5)
    )

    # Boxplot Dunas (Deslocado para baixo)
    bp_duna = ax_box.boxplot(
        dados_duna,
        positions=y_positions - offset,
        flierprops=flierprops,
        vert=False, widths=box_width, patch_artist=True, showfliers=True,
        boxprops=dict(facecolor='white', color=cor_duna, linewidth=1.5),
        whiskerprops=dict(color=cor_duna, linewidth=1.5),
        capprops=dict(color=cor_duna, linewidth=1.5),
        medianprops=dict(color=cor_duna, linewidth=1.5)
    )

    # Ajustes do Eixo Y (Posicionamento e Siglas)
    ax_box.set_yticks(y_positions)
    ax_box.set_yticklabels([siglas_map[r] for r in regioes_ordem], fontsize=10, fontweight='bold')
    ax_box.invert_yaxis()  # Mantém o padrão (S no topo)
    
    ax_box.set_xlim(0, 50)
    ax_box.set_xlabel('Proportion of $PM_{10}$ from DIW (%)', fontsize=10)
    ax_box.grid(axis='x', alpha=0.3, linestyle='--')

    # Legenda customizada
    legend_elements = [
        Patch(facecolor='white', edgecolor=cor_mine, label='Mining', linewidth=1.5),
        Patch(facecolor='white', edgecolor=cor_duna, label='Dune and sandy area', linewidth=1.5)
    ]
    ax_box.legend(handles=legend_elements, loc='lower right', framealpha=0.8, fontsize=9)
    ax_box.set_title('(b)', fontsize=12, loc='left')

    plt.tight_layout()
    if salvar:
        fig.savefig(OUTPUT_DIR + r"figura2_razao_wbd_solos.png", dpi=150, bbox_inches='tight')
    plt.show()

    estatisticas = [
        'Quantidade dados',
        'Máximo',
        'Whisker sup.',
        'Percentil 75',
        'Mediana',
        'Percentil 25',
        'Whisker inf.',
        'Mínimo'
    ]

    # Cabeçalho igual ao da figura
    colunas = pd.MultiIndex.from_tuples([
        ('Mineração', 'S'),
        ('Mineração', 'SE'),
        ('Mineração', 'CO'),
        ('Mineração', 'NE'),
        ('Mineração', 'N'),
        ('Dunas', 'S'),
        ('Dunas', 'SE'),
        ('Dunas', 'NE'),
        ('Dunas', 'N'),
    ])

    tabela = pd.DataFrame(
        index=estatisticas,
        columns=colunas,
        dtype=float
    )

    map_reg = {
        'South': 'S',
        'Southeast': 'SE',
        'Midwest': 'CO',
        'Northeast': 'NE',
        'North': 'N'
    }

    # ---------------- Mineração ----------------
    for i, regiao in enumerate(regioes_ordem):
        sigla = map_reg[regiao]
        stats = estatisticas_boxplot(dados_mine[i])
        tabela[('Mineração', sigla)] = stats

    # ---------------- Dunas ----------------
    for i, regiao in enumerate(regioes_ordem):
        sigla = map_reg[regiao]
        if sigla == 'CO':
            continue
        stats = estatisticas_boxplot(dados_duna[i])
        tabela[('Dunas', sigla)] = stats

    tabela.to_csv('/home/lcqar/BRAIN/emis/windBlowDustBR/Outputs/BR_12km/tables/obj_03_razao_solos.csv')


def plot_painel_concentracao_razao(ds_all_mp10, ds_wbd_mp10, lat, lon, shp, shp2, salvar=True):
    """
    Cria a Figura 3 com 6 subplots (2 linhas x 3 colunas):
 
    Linha 1 (mapas, com fundo recortado por shp2):
        Col 1 -> Concentração média de PM10 (ALL)
        Col 2 -> Concentração média de PM10 (WBD)
        Col 3 -> Razão percentual WBD/ALL
 
    Linha 2 (boxplots horizontais por região):
        Col 1 -> Distribuição da concentração ALL por região
        Col 2 -> Distribuição da concentração WBD por região
        Col 3 -> Distribuição da razão percentual (%) por região
 
    Parâmetros
    ----------
    ds_all_mp10, ds_wbd_mp10 : xr.DataArray
        Aceita tanto a série temporal (com dimensão TSTEP) quanto o campo
        já médio no tempo (ex.: ds_all_10_mes / ds_wbd_10_mes). Caso possuam
        TSTEP, a média temporal é calculada internamente para gerar os mapas
        e os boxplots.
    lat, lon : np.ndarray (2D)
        Coordenadas dos pixels da grade.
    shp : GeoDataFrame
        Shapefile por UF (usado para recortar os pixels por região via SIGLA_UF).
    shp2 : GeoDataFrame
        Shapefile dissolvido por região (usado como fundo dos mapas).
    """
    print("[INFO] Gerando Figura 3 - Painel ALL x WBD x Razão por Região...")
 
    # 1. Garante campo 2D (ROW, COL) médio no tempo, caso ainda haja TSTEP
    ds_all_2d = ds_all_mp10.mean(dim='TSTEP') if 'TSTEP' in ds_all_mp10.dims else ds_all_mp10
    ds_wbd_2d = ds_wbd_mp10.mean(dim='TSTEP') if 'TSTEP' in ds_wbd_mp10.dims else ds_wbd_mp10
 
    # 2. Razão percentual WBD/ALL (mesma lógica usada em plot_razao_wbd_solo)
    razao = (ds_wbd_2d / ds_all_2d) * 100
    razao = xr.where(~np.isfinite(razao), 0, razao)
 
    # 3. GeoDataFrame de pixels com os três campos já "achatados" (1D)
    gdf_pixels = _build_gdf(lat.flatten(), lon.flatten())
    gdf_plot = gdf_pixels.copy()
    gdf_plot['all']   = ds_all_2d.values.flatten()
    gdf_plot['wbd']   = ds_wbd_2d.values.flatten()
    gdf_plot['razao'] = razao.values.flatten()
 
    # 4. Cria a figura: 2 linhas x 3 colunas
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), gridspec_kw={'height_ratios': [3, 1]})
    (ax_map_all, ax_map_wbd, ax_map_razao) = axes[0]
    (ax_box_all, ax_box_wbd, ax_box_razao) = axes[1]
 
    # ------------------------------------------------------------------
    # LINHA 1 - MAPAS (recortados por shp2)
    # ------------------------------------------------------------------
    def _plot_mapa(ax, shp, shp2, coluna, titulo, cmap, cbar_label, vmin=0, vmax=None):
        gdf_mapa = gdf_plot[gdf_plot[coluna] > 0]

        if vmax is None:
            vmax = np.nanpercentile(gdf_mapa[coluna], 98) if len(gdf_mapa) > 0 else 1
 
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        print(1)
        gdf_mapa = gpd.sjoin(gdf_mapa, shp, how="inner", predicate="intersects")
        print(1)
        
        gdf_mapa.plot(column=coluna, cmap=cmap, norm=norm, ax=ax, legend=False, alpha=1)
        shp2.plot(ax=ax, edgecolor='white', color='gainsboro', linewidth=0.5, alpha=0)
 
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cax = inset_axes(ax, width="4%", height="36%", loc='lower left',
                          bbox_to_anchor=(0.15, 0.05, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
        cbar = plt.colorbar(sm, cax=cax, orientation='vertical')
        cbar.set_label(cbar_label, fontsize=12)
        cbar.ax.tick_params(labelsize=12)
        cbar.ax.yaxis.set_label_position('left')
 
        ax.set_xlim(-74, -34); ax.set_ylim(-34, 6)
        ax.set_aspect('equal')
        ax.set_axis_off()
        ax.set_title(titulo, fontsize=20, loc='left')
 
    cmap_conc = plt.get_cmap('viridis')
    cmap_razao = plt.get_cmap('tab20b')
 
    _plot_mapa(ax_map_all, shp, shp2, 'all',   '(a)' , cmap_conc,  r'PM$_{{10}}$ ($\mu g/m^3$)')
    _plot_mapa(ax_map_wbd, shp, shp2, 'wbd',   '(b)',  cmap_conc,  r'PM$_{{10}}$ ($\mu g/m^3$)')
    _plot_mapa(ax_map_razao,shp,shp2, 'razao', '(c)',  cmap_razao, 'Contribution of DIW (%)', vmin=0, vmax=100)
 
    # ------------------------------------------------------------------
    # LINHA 2 - BOXPLOTS VERTICAIS POR REGIÃO
    # ------------------------------------------------------------------
    regioes_ordem = ['North', 'Northeast', 'Midwest', 'Southeast', 'South']
    siglas_map = {'South': 'S', 'Southeast': 'SE', 'Midwest': 'CO', 'Northeast': 'NE', 'North': 'N'}
    x_positions = np.arange(len(regioes_ordem)) # Trocado para x_positions

    dados_all, dados_wbd, dados_razao = [], [], []

    for regiao in regioes_ordem:
        estados = REGIOES[regiao]['estados']
        shp_reg = shp[shp['SIGLA_UF'].isin(estados)]

        gdf_reg = gpd.sjoin(
            gdf_plot, shp_reg[['geometry']],
            predicate='intersects', how='inner'
        ).drop(columns='index_right', errors='ignore')

        v_all   = gdf_reg['all'].dropna().values;   v_all   = v_all[v_all > 0]
        v_wbd   = gdf_reg['wbd'].dropna().values;   v_wbd   = v_wbd[v_wbd > 0]
        v_razao = gdf_reg['razao'].dropna().values; v_razao = v_razao[v_razao > 0]

        dados_all.append(v_all if len(v_all) > 0 else np.array([np.nan]))
        dados_wbd.append(v_wbd if len(v_wbd) > 0 else np.array([np.nan]))
        dados_razao.append(v_razao if len(v_razao) > 0 else np.array([np.nan]))

    def _plot_boxplot(ax, dados, cor, ylabel, titulo, ylim=None, log_y=False):
        flierprops = dict(marker='o', markersize=2, linestyle='none')

        ax.boxplot(
            dados, positions=x_positions, vert=True, widths=0.5, flierprops=flierprops,
            patch_artist=True, showfliers=True,
            boxprops=dict(facecolor='white', color=cor, linewidth=1.5),
            whiskerprops=dict(color=cor, linewidth=1.5),
            capprops=dict(color=cor, linewidth=1.5),
            medianprops=dict(color=cor, linewidth=1.5)
        )
        # Configurações do eixo X (Regiões)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([siglas_map[r] for r in regioes_ordem], fontsize=15, fontweight='bold')
        
        # Configurações do eixo Y (Concentração/Razão)
        ax.set_ylabel(ylabel, fontsize=15)
        
        # Aplica escala logarítmica se solicitado
        if log_y:
            ax.set_yscale('log')
            
        if ylim is not None:
            ax.set_ylim(*ylim)
            
        ax.grid(axis='y', alpha=0.3, linestyle='--') # Grade no eixo Y

        ax.set_title(titulo, fontsize=20, loc='left')

    cor_all, cor_wbd, cor_razao = '#3498db', '#e74c3c', '#9b59b6'

    # Plota os gráficos ALL e WBD com escala log no eixo Y
    _plot_boxplot(ax_box_all,  dados_all,  cor_all,  r'PM$_{{10}}$ ($\mu g/m^3$)', '(d)', log_y=True)
    _plot_boxplot(ax_box_wbd,  dados_wbd,  cor_wbd,  r'PM$_{{10}}$ ($\mu g/m^3$)', '(e)', log_y=True)

    # Sincroniza o eixo Y entre ax_box_all e ax_box_wbd, forçando o mínimo a ser 0.01
    _, y1_max = ax_box_all.get_ylim()
    _, y2_max = ax_box_wbd.get_ylim()
    y_max_comum = max(y1_max, y2_max)
    
    # Define o limite ymin = 0.01 e o ymax compartilhado
    ax_box_all.set_ylim(0.01, y_max_comum)
    ax_box_wbd.set_ylim(0.01, y_max_comum)

    # Plota o gráfico de Razão separadamente com limite de 0 a 15%
    _plot_boxplot(ax_box_razao, dados_razao, cor_razao, 'Contribution of DIW (%)', '(f)',ylim=(0, 15))

    plt.tight_layout()

    if salvar:
        fig.savefig(OUTPUT_DIR + r"figura3_painel_all_wbd_razao.png", dpi=150, bbox_inches='tight')
    plt.show()
 
    def salvar_tabela_estatisticas(dados, nome_arquivo):

        estatisticas = [
            'Quantidade dados',
            'Máximo',
            'Whisker sup.',
            'Percentil 75',
            'Mediana',
            'Percentil 25',
            'Whisker inf.',
            'Mínimo'
        ]

        colunas = ['S', 'SE', 'CO', 'NE', 'N']

        tabela = pd.DataFrame(
            index=estatisticas,
            columns=colunas,
            dtype=float
        )

        map_reg = {
            'South':'S',
            'Southeast':'SE',
            'Midwest':'CO',
            'Northeast':'NE',
            'North':'N'
        }

        for i, regiao in enumerate(regioes_ordem):
            sigla = map_reg[regiao]
            stats = estatisticas_boxplot(dados[i])
            tabela[sigla] = stats

        tabela = tabela.round(3)

        caminho = (
            '/home/lcqar/BRAIN/emis/windBlowDustBR/'
            'Outputs/BR_12km/tables/'
            + nome_arquivo
        )

        tabela.to_csv(caminho)
        print(f'[INFO] Tabela salva: {caminho}')


    # Salva as três tabelas
    salvar_tabela_estatisticas(
        dados_all,
        'obj_03_all.csv'
    )

    salvar_tabela_estatisticas(
        dados_wbd,
        'obj_03_wbd.csv'
    )

    salvar_tabela_estatisticas(
        dados_razao,
        'obj_03_razao.csv'
    )
 
    return fig

'''plot_impacto_quadrantes(
    ds_all_10_dia=ds_wbd_10_dia, 
    ds_emis_10_dia=ds_emis_10_dia, 
    lat=lat, lon=lon, shp=shp, 
    PI_MP10=PF_MP10
)'''

plot_razao_wbd_solo(
    ds_wbd_10_mes=ds_wbd_10_mes, 
    ds_all_10_mes=ds_all_10_mes, 
    lat=lat, lon=lon, shp=shp
)

plot_painel_concentracao_razao(
    ds_all_mp10=ds_all_mp10_int,
    ds_wbd_mp10=ds_wbd_mp10_int,
    lat=lat, lon=lon, shp=shp, shp2=shp2
)
