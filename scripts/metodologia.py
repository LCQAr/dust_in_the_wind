# -*- coding: utf-8 -*-
"""
Mapa do Brasil por região + áreas de Duna/Areal/Praia e Mineração.
 
Figura com 1 linha e 3 colunas:
    Col 1 -> Brasil dividido por região (cores iguais às do objetivo_02.py)
    Col 2 -> Pixels onde content_duna > 0  ("Dunas / Areais / Praias")
    Col 3 -> Pixels onde content_mine > 0  ("Áreas de mineração")
 
A obtenção de content_duna e content_mine segue exatamente a mesma lógica
da função `_uso_solo` do objetivo_02.py: lê as grades via gridDetails (grd)
e regridMAPBIOMAS (regMap) e calcula a fração de área de cada pixel ocupada
por duna/areal/praia e por mineração.
"""
 
import sys
import os
 
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pyproj
import xarray as xr
from shapely.geometry import Point, box
import matplotlib.patheffects as path_effects
# Garante que os módulos auxiliares (gridDetails, regridMAPBIOMAS) sejam
# encontrados mesmo que o script seja executado de outro diretório.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import regridMAPBIOMAS as regMap
import gridDetails as grd
 
# ──────────────────────────────────────────────────────────────────────────────
# Configuração de caminhos (ajuste conforme necessário)
# ──────────────────────────────────────────────────────────────────────────────
 
OUTPUT_DIR = "/home/lcqar/BRAIN/emis/windBlowDustBR/Outputs/BR_12km/mapas_regioes_uso_solo/"
SHP_PATH   = "/home/lcqar/BRAIN/emis/windBlowDustBR/inputs/shapefiles/BR_UF_2024.shp"
 
NC_EXEMPLO = (
    "/home/lcqar/BRAIN/emis/windBlowDustBR/Outputs/BR_12km/"
    "windBlowDust_PM10_2023-02-24-00:00:00_2023-02-25-00:00:00.nc"
)
 
# ──────────────────────────────────────────────────────────────────────────────
# Constantes / mapeamentos (idênticos ao objetivo_02.py)
# ──────────────────────────────────────────────────────────────────────────────
 
REGIOES = {
    'South':          {'cor': '#E27668', 'estados': ['RS', 'SC', 'PR']},
    'Southeast':      {'cor': '#75899A', 'estados': ['SP', 'MG', 'RJ', 'ES']},
    'Midwest': {'cor': '#7D5C65', 'estados': ['MT', 'MS', 'GO', 'DF']},
    'Northeast':     {'cor': '#C89D3C', 'estados': ['BA', 'SE', 'AL', 'PE', 'PB', 'RN', 'CE', 'PI', 'MA']},
    'North':        {'cor': '#2C5E4C', 'estados': ['TO', 'PA', 'AP', 'RR', 'AM', 'AC', 'RO']},
}
 
# Mapa UF -> Região, usado para dissolver o shapefile por região
regioes = {
    'RS': 'South', 'SC': 'South', 'PR': 'South',
    'SP': 'Southeast', 'RJ': 'Southeast', 'MG': 'Southeast', 'ES': 'Southeast',
    'MS': 'Midwest', 'MT': 'Midwest', 'GO': 'Midwest', 'DF': 'Midwest',
    'BA': 'Northeast', 'SE': 'Northeast', 'AL': 'Northeast', 'PE': 'Northeast',
    'PB': 'Northeast', 'RN': 'Northeast', 'CE': 'Northeast', 'PI': 'Northeast', 'MA': 'Northeast',
    'AC': 'North', 'AM': 'North', 'RR': 'North', 'RO': 'North',
    'PA': 'North', 'AP': 'North', 'TO': 'North',
}
 
# Cores de uso do solo (mesmas usadas no objetivo_02.py / objetivo03.py)
COR_DUNA = '#D2A679'   # caramelo
COR_MINE = '#808080'   # cinza
 
# ──────────────────────────────────────────────────────────────────────────────
# Helpers geoespaciais (idênticos ao objetivo_02.py)
# ──────────────────────────────────────────────────────────────────────────────
 
def _point_to_square(point, tamanho=5750):
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
 
 
def ioapiCoords(ds):
    lonI = ds.XORIG
    latI = ds.YORIG
 
    xcell = ds.XCELL
    ycell = ds.YCELL
    ncols = ds.NCOLS
    nrows = ds.NROWS
 
    lon = np.arange(lonI, (lonI + ncols * xcell), xcell)
    lat = np.arange(latI, (latI + nrows * ycell), ycell)
 
    xv, yv = np.meshgrid(lon, lat)
    return xv, yv, lon, lat
 
 
def eqmerc2latlon(ds, xv, yv):
    mapstr = '+proj=merc +a=%s +b=%s +lat_ts=0 +lon_0=%s' % (
        6370000, 6370000, ds.XCENT)
    p = pyproj.Proj(mapstr)
    xlon, ylat = p(xv, yv, inverse=True)
    return xlon, ylat
 
 
def latlon_2d(dir_data):
    """Converte as coordenadas do NetCDF (grade IOAPI) para arrays 2D lat/lon."""
    data = xr.open_dataset(dir_data)
    xv, yv, lon, lat = ioapiCoords(data)
    xlon, ylat = eqmerc2latlon(data, xv, yv)
    lon2d = xlon.flatten()
    lat2d = ylat.flatten()
    return lon2d, lat2d
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Conteúdo de uso do solo (mesma lógica de `_uso_solo` do objetivo_02.py)
# ──────────────────────────────────────────────────────────────────────────────
 
def obter_content_uso_solo():
    """
    Lê as grades do WRF/MCIP e do MapBiomas (via gridDetails / regridMAPBIOMAS)
    e calcula a fração de área de cada pixel ocupada por Duna/Areal/Praia e
    por Mineração, seguindo exatamente a mesma lógica de `_uso_solo` do
    objetivo_02.py.
 
    Retorno
    -------
    content_duna : np.ndarray (n_pixels,)
        Fração [0, 1] de duna/areal/praia em relação à área (duna + mineração).
    content_mine : np.ndarray (n_pixels,)
        Fração [0, 1] de mineração em relação à área (duna + mineração).
    """
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
 
    # alarea[0] = duna, alarea[1] = mineração (área em cada pixel)
    area_duna  = alarea[0, :, :].flatten()
    area_mine  = alarea[1, :, :].flatten()
    area_total = area_duna + area_mine
 
    with np.errstate(invalid='ignore', divide='ignore'):
        content_duna = np.where(area_total > 0, area_duna / area_total, 0.0)
        content_mine = np.where(area_total > 0, area_mine / area_total, 0.0)
 
    print(f"[obter_content_uso_solo] content_duna max={content_duna.max():.4f}  "
          f"pixels>0: {(content_duna > 0).sum()}")
    print(f"[obter_content_uso_solo] content_mine max={content_mine.max():.4f}  "
          f"pixels>0: {(content_mine > 0).sum()}")
 
    return content_duna, content_mine
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Figura principal: 1 linha x 3 colunas
# ──────────────────────────────────────────────────────────────────────────────
 
def plot_mapa_regioes_uso_solo(
    lat, lon, shp, content_duna, content_mine,
    salvar=True, mostrar=True
):
    """
    Cria a figura com 3 colunas:
      Col 1 -> Brasil dividido por região, cada região na cor de REGIOES.
      Col 2 -> Pixels com content_duna > 0, título "Dunas / Areais / Praias",
               com o Brasil em gainsboro dividido por região no fundo.
      Col 3 -> Pixels com content_mine > 0, título "Áreas de mineração",
               com o Brasil em gainsboro dividido por região no fundo.
    """
    print("[INFO] Gerando figura - Regiões x Dunas/Areais x Mineração...")
 
    # 1. GeoDataFrame de pixels com os pesos de uso do solo
    gdf_pixels = _build_gdf(lat.flatten(), lon.flatten())
    gdf_pixels['content_duna'] = content_duna
    gdf_pixels['content_mine'] = content_mine
 
    # 2. Shapefile dissolvido por região (fundo cinza dos painéis 2 e 3)
    shp2 = shp.copy()
    shp2['REGIAO'] = shp2['SIGLA_UF'].map(regioes)
    shp2 = shp2.dissolve(by='REGIAO').reset_index()
 
    xlim, ylim = (-74, -34), (-34, 6)
 
    # 3. Figura
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    ax_regioes, ax_duna, ax_mine = axes
 
    # ------------------------------------------------------------------
    # COLUNA 1 - Brasil dividido por região
    # ------------------------------------------------------------------
    for regiao, info in REGIOES.items():
        shp_reg = shp2[shp2['REGIAO'] == regiao]
        shp_reg.plot(ax=ax_regioes, color=info['cor'], edgecolor='white', linewidth=0.6, alpha=0.7)
 
    ax_regioes.set_xlim(*xlim); ax_regioes.set_ylim(*ylim)
    ax_regioes.set_aspect('equal')
    ax_regioes.set_axis_off()
    ax_regioes.set_title('(a)', fontsize=15, loc='left')
 
    legend_regioes = [
        mpatches.Patch(facecolor=info['cor'], edgecolor='white', label=regiao)
        for regiao, info in REGIOES.items()
    ]
    ax_regioes.legend(handles=legend_regioes, loc='lower left', framealpha=0.8, fontsize=12)
 
    # ------------------------------------------------------------------
    # COLUNA 2 - Dunas / Areais / Praias (content_duna > 0)
    # ------------------------------------------------------------------
    gdf_duna = gdf_pixels[gdf_pixels['content_duna'] > 0]
 
    shp2.plot(ax=ax_duna, edgecolor='white', color='gainsboro', linewidth=0.5, alpha=0.7)
    gdf_duna.plot(ax=ax_duna, color=COR_DUNA, edgecolor='none')
 
    ax_duna.set_xlim(*xlim); ax_duna.set_ylim(*ylim)
    ax_duna.set_aspect('equal')
    ax_duna.set_axis_off()
    ax_duna.set_title('(b)', fontsize=15, loc='left')
    ax_duna.legend(
        handles=[mpatches.Patch(facecolor=COR_DUNA, edgecolor='none', label='Dune / Sandy area')],
        loc='lower left', framealpha=0.8, fontsize=12
    )
 
    # ------------------------------------------------------------------
    # COLUNA 3 - Áreas de mineração (content_mine > 0)
    # ------------------------------------------------------------------
    gdf_mine = gdf_pixels[gdf_pixels['content_mine'] > 0]
 
    shp2.plot(ax=ax_mine, edgecolor='white', color='gainsboro', linewidth=0.5, alpha=0.7)
    gdf_mine.plot(ax=ax_mine, color=COR_MINE, edgecolor='none')
 
    ax_mine.set_xlim(*xlim); ax_mine.set_ylim(*ylim)
    ax_mine.set_aspect('equal')
    ax_mine.set_axis_off()
    ax_mine.set_title('(c)', fontsize=15, loc='left')
    ax_mine.legend(
        handles=[mpatches.Patch(facecolor=COR_MINE, edgecolor='none', label='Mining')],
        loc='lower left', framealpha=0.8, fontsize=12
    )

    # ------------------------------------------------------------------
    # ADICIONAR SIGLAS NO CENTROIDE DE CADA REGIÃO (Nos 3 mapas)
    # ------------------------------------------------------------------
    siglas_regioes = {'South': 'S', 'Southeast': 'SE', 'Midwest': 'CO', 'Northeast': 'NE', 'North': 'N'}
    
    for _, row in shp2.iterrows():
        regiao = row['REGIAO']
        if regiao in siglas_regioes:
            centroid = row['geometry'].centroid
            sigla = siglas_regioes[regiao]
            
            # Texto na Coluna 1 (Preto com contorno branco para contrastar com os mapas coloridos)
            t1 = ax_regioes.text(centroid.x, centroid.y, sigla, fontsize=12, fontweight='bold',
                                 ha='center', va='center', color='black')
            t1.set_path_effects([path_effects.withStroke(linewidth=2, foreground='white')])
            
    plt.tight_layout()
 
    if salvar:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        nome = "mapa_regioes_duna_mineracao.png"
        fig.savefig(OUTPUT_DIR + nome, dpi=150, bbox_inches='tight')
        print(f"[INFO] Figura salva em: {OUTPUT_DIR + nome}")
    if mostrar:
        plt.show()
 
    return fig
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Bloco de execução
# ──────────────────────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
 
    print("[INFO] Lendo coordenadas...")
    lon, lat = latlon_2d(NC_EXEMPLO)
 
    print("[INFO] Lendo shapefile...")
    shp = gpd.read_file(SHP_PATH, engine="pyogrio")
 
    print("[INFO] Calculando content_duna e content_mine...")
    content_duna, content_mine = obter_content_uso_solo()
 
    plot_mapa_regioes_uso_solo(
        lat=lat, lon=lon, shp=shp,
        content_duna=content_duna, content_mine=content_mine,
        salvar=True, mostrar=False
    )
 
    print("[INFO] Concluído.")
