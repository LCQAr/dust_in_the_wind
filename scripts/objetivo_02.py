# -*- coding: utf-8 -*-
"""
Created on Tue May 12 08:24:01 2026

@author: José
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import calendar
import pickle
import pyproj
import xarray as xr
import regridMAPBIOMAS as regMap
import gridDetails as grd
from matplotlib.colors import LinearSegmentedColormap
from shapely.geometry import Point, box
from matplotlib.colors import ListedColormap, LogNorm
from scipy.stats import kruskal
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
 
# ──────────────────────────────────────────────────────────────────────────────
# Configuração de caminhos (ajuste conforme necessário)
# ──────────────────────────────────────────────────────────────────────────────
 
OUTPUT_DIR = "/home/lcqar/BRAIN/emis/windBlowDustBR/Outputs/BR_12km/objetivo02/"
PKL_PATH   = OUTPUT_DIR + "emissoes_wbd.pkl"
SHP_PATH   = "/home/lcqar/BRAIN/emis/windBlowDustBR/inputs/shapefiles/BR_UF_2024.shp"
 
# ──────────────────────────────────────────────────────────────────────────────
# Constantes / mapeamentos
# ──────────────────────────────────────────────────────────────────────────────
'''
REGIOES = {
    'Sul':          {'cor': '#E27668', 'estados': ['RS', 'SC', 'PR']},           # vermelho
    'Sudeste':      {'cor': '#75899A', 'estados': ['SP', 'MG', 'RJ', 'ES']},     # azul
    'Centro-Oeste': {'cor': '#9b59b6', 'estados': ['MT', 'MS', 'GO', 'DF']},     # roxo
    'Nordeste':     {'cor': '#f1c40f', 'estados': ['BA', 'SE', 'AL', 'PE', 'PB', 'RN', 'CE', 'PI', 'MA']},  # amarelo
    'Norte':        {'cor': '#2ecc71', 'estados': ['TO', 'PA', 'AP', 'RR', 'AM', 'AC', 'RO']},              # verde
}'''
REGIOES = {
    'South':          {'cor': '#E27668', 'estados': ['RS', 'SC', 'PR']},            # rosa antigo / bordô claro
    'Southeast':      {'cor': '#75899A', 'estados': ['SP', 'MG', 'RJ', 'ES']},      # azul marinho acinzentado
    'Midwest': {'cor': '#7D5C65', 'estados': ['MT', 'MS', 'GO', 'DF']},      # malva escuro
    'Northeast':     {'cor': '#C89D3C', 'estados': ['BA', 'SE', 'AL', 'PE', 'PB', 'RN', 'CE', 'PI', 'MA']},  # dourado queimado
    'North':        {'cor': '#2C5E4C', 'estados': ['TO', 'PA', 'AP', 'RR', 'AM', 'AC', 'RO']},              # verde floresta
}
 
# Ordem padronizada de estados por região (Sul → Norte geográfico)
ESTADOS_ORDEM = ['RS','SC','PR','SP','MG','RJ','ES',
                 'BA','SE','AL','PE','PB','RN','CE','PI','MA',
                 'TO','PA','AP','RR','AM','AC','RO',
                 'MT','MS','GO','DF']

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

'''
CORES_MENSAIS = [
    "darkred","red","orange","lightblue","paleturquoise","lightcyan",
    "dodgerblue","blue","navy","lightsalmon","peachpuff","lightpink"
]'''
 
CORES_MENSAIS = [
    # GRUPO 1: Tons Quentes e Vibrantes (Jan, Fev, Mar) - Alto contraste contra o cinza
    '#FF4500',  # Jan (laranja-avermelhado vibrante)
    '#FF8C00',  # Fev (laranja escuro)
    '#FFBF00',  # Mar (âmbar/amarelo seletivo)

    # GRUPO 2: Tons de Roxo e Violeta (Abr, Mai, Jun) - Elegante e visível
    '#6A0DAD',  # Abr (roxo rico)
    '#8A2BE2',  # Mai (azul-violeta)
    '#BA55D3',  # Jun (orquídea média)

    # GRUPO 3: Tons de Azul Claros e Puros (Jul, Ago, Set) - Frio e limpo
    '#1E90FF',  # Jul (azul esquivo)
    '#00BFFF',  # Ago (azul celeste profundo)
    '#87CEFA',  # Set (azul céu claro)

    # GRUPO 4: Tons de Verde Naturais (Out, Nov, Dez) - Equilibrado
    '#006400',  # Out (verde escuro)
    '#228B22',  # Nov (verde floresta)
    '#32CD32',  # Dez (verde lima)
]
 
LABELS_MENSAIS  = LABELS_MENSAIS  = ['Jan','Feb','Mar','Abr','Mai','Jun',
                   'Jul','Ago','Set','Out','Nov','Dez']

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
 
# ──────────────────────────────────────────────────────────────────────────────
# Helpers geoespaciais
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

    elif tipo == 'pixel_min':
        # Índice do período com maior emissão (0-based)
        result = np.argmin(vals, axis=1).astype(float)
        # Pixels com todos os valores iguais (ex: zero) → NaN
        result[np.all(vals == vals[:, [0]], axis=1)] = np.nan
        return pd.Series(result, index=pixel_df.index)
 
    elif tipo == 'amplitude':
        monthly_means = vals*3600*1e-6                        # já é soma por período
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
    -------
    pixel_df_pond : DataFrame  (n_pixels × n_periodos)
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
 
def _plot_mapa(ax, gdf_metrica, shp, escala_espacial, tipo, freq, metrica_vals):
    """
    Pinta os pixels no eixo `ax`.
 
    - tipo == 'pixel_max': cores discretas (mês ou hora)
    - outros tipos: escala contínua (viridis / plasma)
    """
    gdf_plot = gdf_metrica.copy()
    gdf_plot['metrica'] = metrica_vals.values
 
 
    # Determina limites do mapa
    if escala_espacial in ('Brasil', 'Solo Duna', 'Solo Mineração'):
        xlim, ylim = (-74, -34), (-34, 6)
        shp2=shp
        shp2['REGIAO']=shp2['SIGLA_UF'].map(regioes)
        shp2=shp2.dissolve(by='REGIAO').reset_index()
    else:
        ufs_regiao = REGIOES[escala_espacial]['estados']
        shp2 = shp[shp['SIGLA_UF'].isin(ufs_regiao)].copy() 
        if escala_espacial == 'Nordeste':
            xlim, ylim = (-54, -34), (-20, 0)
        elif escala_espacial == 'Norte':
            xlim, ylim = (-74, -44), (-20, 10)
        elif escala_espacial == 'Sul':
            xlim, ylim = (-61, -46), (-35, -20) 
        elif escala_espacial == 'Sudeste':
            xlim, ylim = (-55, -35), (-30, -10) 
        elif escala_espacial == 'Centro-Oeste':
            xlim, ylim = (-68, -43), (-30, -5)
 
    if tipo == 'pixel_max' or tipo == 'pixel_min':
        # Colormap discreto
        if freq == 'monthly':
            cmap = ListedColormap(CORES_MENSAIS)
            n_cat = 12
            tick_labels = LABELS_MENSAIS
        else:
            cmap = plt.get_cmap('hsv', 24)
            n_cat = 24
            tick_labels = LABELS_HORARIOS
 
        norm = mcolors.BoundaryNorm(boundaries=np.arange(-0.5, n_cat + 0.5, 1), ncolors=n_cat)
        
        gdf_plot_valid = gdf_plot.dropna(subset=['metrica'])
        shp2.plot(ax=ax, edgecolor='white', color='gainsboro', linewidth=0.5, alpha=0.7)
        gdf_plot_valid.plot(column='metrica', cmap=cmap, norm=norm, ax=ax, legend=False, alpha=1)
 
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
    
        cax = inset_axes(
            ax,
            width="3%",          # largura da colorbar
            height="36%",        # altura da colorbar
            loc='lower left',
            bbox_to_anchor=(0.1, 0.05, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0
        )
        
        cbar = plt.colorbar(sm, cax=cax, orientation='vertical')
        if freq == 'hourly':
            # Pula de 3 em 3 (0, 3, 6, 9...)
            ticks_pos = np.arange(0, n_cat, 3) 
            # Seleciona os labels correspondentes a essas posições
            labels_selecionados = [tick_labels[i] for i in ticks_pos]
            cbar.set_ticks(ticks_pos)
            cbar.set_ticklabels(labels_selecionados, fontsize=7)
        else:
            cbar.set_ticks(np.arange(n_cat)) 
            cbar.set_ticklabels(tick_labels, fontsize=7)

        # Configurações opcionais
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label('Peak emission period', fontsize=9)
        cbar.ax.yaxis.set_label_position('left')
 
    else:
        # Colormap contínuo
        gdf_plot_valid = gdf_plot.dropna(subset=['metrica'])
        vmin = gdf_plot_valid[gdf_plot_valid['metrica']>0]['metrica'].quantile(0)
        vmax = gdf_plot_valid['metrica'].quantile(1)
        if tipo == 'msi':
            vmax = 1
        elif tipo == 'amplitude':
            vmin = 0.001
        #vmin = max(vmin, 1e-10)  # evita log de zero
 
        #if vmax <= vmin:
        #    vmax = vmin * 10.0

        cmap_nome = {'Magnitude': 'copper_r', 'amplitude': 'Purples', 'forca': 'Reds', 'msi': 'copper_r'}.get(tipo, 'viridis')

        if tipo == 'msi':
            vmax = 1
            
            # Cria um degradê contínuo estilo 'spring', mas com tons desaturados e elegantes
            # Rosa magenta antigo -> Amarelo dourado suave
            cores_spring_suave = ['#FF5395','#FDC939'] 
            
            cmap = LinearSegmentedColormap.from_list('spring_suave', cores_spring_suave)
            cmap = plt.get_cmap('cool')
            norm = mcolors.Normalize(vmin=0, vmax=1)

        # Usa apenas a metade superior do colormap (0.5 → 1.0)
        elif tipo == 'amplitude':
            base_cmap = plt.get_cmap('bone_r')
            cmap = LinearSegmentedColormap.from_list(
                'Purples_half',
                base_cmap(np.linspace(0.3, 1, 256))
            )
        else:
            cmap = plt.get_cmap(cmap_nome)
 
        try:
            if tipo == 'amplitude':
                norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
            elif tipo == 'Magnitude':
                norm = mcolors.LogNorm(vmin=0.01, vmax=vmax)
            elif tipo == 'msi':
                norm = mcolors.Normalize(vmin=0, vmax=1)
            else:
                norm = mcolors.Normalize(vmin=0, vmax=vmax)
        except Exception:
            norm = mcolors.Normalize(vmin=0, vmax=vmax)
 
        shp2.plot(ax=ax, edgecolor='white', color='gainsboro', linewidth=0.5, alpha=0.7)
        gdf_plot_valid = gdf_plot.query('metrica > 0')
        gdf_plot_valid.plot(column='metrica', cmap=cmap, norm=norm, ax=ax, legend=False, alpha=1)
 
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
    
        cax = inset_axes(
            ax,
            width="3%",          # largura da colorbar
            height="36%",        # altura da colorbar
            loc='lower left',
            bbox_to_anchor=(0.15, 0.05, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0
        )
        
        cbar = plt.colorbar(sm, cax=cax, orientation='vertical')
        
        # Configurações opcionais
        cbar.ax.tick_params(labelsize=7)
        
        label_cbar = {
            'amplitude': 'Amplitude (ton)',
            'forca':     'Força Relativa',
            'msi':       'MSI (Markham Index)',
            'Magnitude': 'PM$_{10}$ emission (ton/ano)'
        }.get(tipo, tipo)
        cbar.set_label(label_cbar, fontsize=9)
        cbar.ax.yaxis.set_label_position('left')
 
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    
    # Remove ticks e rótulos dos eixos
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Remove os títulos dos eixos
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(length=0)
    ax.set(frame_on=False)
    
    titulo_mapa = {
        'pixel_max':  'Período de maior emissão',
        'amplitude':  'Amplitude da sazonalidade',
        'forca':      'Força relativa da sazonalidade',
        'msi':        'Índice de Markham (MSI)',
        'pixel_min':  'Período de menor emissão'
    }.get(tipo, tipo)
    #ax.set_title(titulo_mapa, fontsize=11, fontweight='bold')
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Plot do boxplot (painel direito)
# ──────────────────────────────────────────────────────────────────────────────
 
def _boxplot_por_estado(ax, gdf_pixels, metrica_vals, shp, estados_lista, tipo, escala_espacial, content):
    """Boxplot com estados do eixo X e a métrica no eixo Y."""
    dados_bp = []
    labels_bp = []
 
    gdf_m = gdf_pixels.copy()
    gdf_m['metrica'] = metrica_vals.values
 
    for estado in estados_lista:
        shp_est = shp[shp['SIGLA_UF'] == estado]
        gdf_est = gpd.clip(gdf_m, shp_est)
        if escala_espacial == 'Solo Duna' or escala_espacial == 'Solo Mineração':
            vals = gdf_est['metrica']*content
            vals = vals.dropna().values
        else:
            vals = gdf_est['metrica'].dropna().values
        vals = vals[vals > 0]
        labels_bp.append(f"{estado}\n(n={len(vals)})")
        dados_bp.append(vals)
        print(estado)
        print(vals)
 
    ax.boxplot(dados_bp, labels=labels_bp, showfliers=False, vert=False)
    if tipo == 'pixel_max':
        ax.set_xscale('log')
    ax.set_ylabel('Estado', fontsize=9)
    ax.set_xlabel('Valor', fontsize=9)
    ax.tick_params(axis='y', labelsize=7)
    ax.grid(axis='x', alpha=0.3)
 
def _boxplot_temporal(ax, gdf_pixels, dfs, var, freq, metrica_vals, escala_espacial, shp, tipo, content):
    """
    Boxplot com o período temporal no eixo X e a distribuição dos valores
    de emissão (não a métrica agregada) no eixo Y – para Brasil / solos.
    """
    if freq == 'monthly':
        labels = LABELS_MENSAIS
        periodos = list(range(12))
    else:
        labels = LABELS_HORARIOS
        periodos = list(range(24))
 
    # Máscara espacial
    gdf_m = gdf_pixels.copy()
    gdf_m['metrica'] = metrica_vals.values
 
    if escala_espacial not in ('Brasil', 'Solo Duna', 'Solo Mineração'):
        if escala_espacial in REGIOES:
            estados = REGIOES[escala_espacial]['estados']
        else:
            estados = [escala_espacial]
        shp_sel = shp[shp['SIGLA_UF'].isin(estados)]
        gdf_m = gpd.clip(gdf_m, shp_sel)
 
    idx_validos = gdf_m.index
 
    dados_bp = []
    for t in periodos:
        if t not in dfs[var][freq]:
            dados_bp.append(np.array([]))
            continue
        #print(dfs[var][freq][t])
        #print(content)
        if escala_espacial == 'Solo Duna' or escala_espacial == 'Solo Mineração':
            df_t = dfs[var][freq][t]*content[:, None]
        else:
            df_t = dfs[var][freq][t]
        vals = df_t.loc[df_t.index.isin(idx_validos)].sum(axis=1).values
        vals = vals[vals > 0]
        dados_bp.append(vals)
 
    ax.boxplot(dados_bp, labels=labels, showfliers=False, vert=False)
    if tipo == 'pixel_max':
        ax.set_xscale('log')
    ax.set_ylabel('Período' if freq == 'monthly' else 'Hour (UTC)', fontsize=9)
    ax.set_xlabel('Emissão (g/s)', fontsize=9)
    ax.tick_params(axis='y', labelsize=8)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
 
# ──────────────────────────────────────────────────────────────────────────────
# Helpers: extrai emissões brutas por uso do solo para um conjunto de pixels
# ──────────────────────────────────────────────────────────────────────────────

def _emissoes_por_uso_solo(dfs, var, freq, idx_validos, conteudos_uso_solo):
    """
    Retorna dict {uso_solo: {periodo: array_vals}} com emissões ponderadas
    pelo conteúdo de cada uso do solo.

    Parâmetros
    ----------
    dfs              : dict de emissões
    var              : str ('PM10', 'PMC', 'PMFINE')
    freq             : str ('monthly' | 'hourly')
    idx_validos      : índices dos pixels da região de interesse
    conteudos_uso_solo : dict {nome_uso: array_peso_por_pixel}
    """
    periodos = sorted(dfs[var][freq].keys())
    resultado = {}
    for nome_uso, peso in conteudos_uso_solo.items():
        resultado[nome_uso] = {}
        for t in periodos:
            df_t = dfs[var][freq][t]
            if peso is not None:
                df_t = df_t * peso[:, None]
            vals = df_t.loc[df_t.index.isin(idx_validos)].sum(axis=1).values
            vals = vals[vals > 0]
            resultado[nome_uso][t] = vals
    return resultado, periodos


# ──────────────────────────────────────────────────────────────────────────────
# TIPO 1 – pixel_max: mapa + barras empilhadas por região + por uso do solo
# ──────────────────────────────────────────────────────────────────────────────

def plot_pixel_max(
    dfs, tipo, lat, lon, shp,
    var, escala_temporal,
    salvar=True, mostrar=False
):
    """
    Figura com 3 colunas:
      col 1 – mapa do pixel com maior emissão (igual a _plot_mapa)
      col 2 – barras empilhadas: frequência do mês/hora de maior emissão,
               eixo Y = meses/horas, eixo X = frequência, cores = regiões
      col 3 – barras empilhadas: mesma frequência por uso do solo (duna e mineração),
               calculadas via _uso_solo com pixel_df_duna e pixel_df_mine
    """
    freq = 'monthly' if escala_temporal == 'mensal' else 'hourly'
    n_periodos = 12 if freq == 'monthly' else 24
    labels_per = LABELS_MENSAIS if freq == 'monthly' else LABELS_HORARIOS

    # ── GDF completo do Brasil ────────────────────────────────────────────────
    gdf_pixels = _build_gdf(lat, lon)
    pixel_df   = _pixel_dataframe(dfs, var, freq)

    # ── Ponderação por uso do solo ────────────────────────────────────────────
    pixel_df, content_duna = _uso_solo(pixel_df, var, 'Solo Duna')
    pixel_df, content_mine = _uso_solo(pixel_df, var, 'Solo Mineração')

    pixel_df_duna = pixel_df * content_duna[:, None]
    pixel_df_mine = pixel_df * content_mine[:, None]

    # ── Métrica pixel_max para o Brasil (sem ponderação, para o mapa) ─────────
    pixel_df_base = _pixel_dataframe(dfs, var, freq)
    metrica_vals  = _calc_metrica(pixel_df_base, tipo, freq)
    gdf_metrica   = gdf_pixels.loc[metrica_vals.index].copy()
    gdf_metrica['metrica'] = metrica_vals.values

    # ── pixel_max ponderado por uso do solo ───────────────────────────────────
    metrica_duna = _calc_metrica(pixel_df_duna, tipo, freq)
    metrica_mine = _calc_metrica(pixel_df_mine, tipo, freq)

    # ── Figura ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        1, 3,
        figsize=(10, 5),
        gridspec_kw={'width_ratios': [1.4, 0.6, 0.6], 'wspace': 0.15}
    )

    ax_map, ax_reg, ax_solo = axes

    # Painel 1 – mapa
    _plot_mapa(ax_map, gdf_metrica, shp, 'Brasil', tipo, freq, metrica_vals)
    ax_map.set_title('(a)', fontsize=12,loc='left')

    # ── Frequência por região ─────────────────────────────────────────────────
    freq_regiao = {r: np.zeros(n_periodos) for r in REGIOES}
    gdf_base_valid = gdf_metrica.dropna(subset=['metrica'])
    for regiao, info in REGIOES.items():
        shp_reg = shp[shp['SIGLA_UF'].isin(info['estados'])]
        gdf_reg = gpd.clip(gdf_base_valid, shp_reg)
        for t in range(n_periodos):
            freq_regiao[regiao][t] = (gdf_reg['metrica'] == t).sum()

    # Painel 2 – barras empilhadas por região
    bottom = np.zeros(n_periodos)
    for regiao, info in REGIOES.items():
        vals = freq_regiao[regiao]
        ax_reg.barh(
            np.arange(n_periodos), vals,
            left=bottom,
            color=info['cor'],
            alpha=0.55,
            edgecolor='white', linewidth=0.3,
            label=regiao
        )
        bottom += vals

    ax_reg.set_yticks(np.arange(n_periodos))
    ax_reg.set_yticklabels(labels_per, fontsize=7)
    ax_reg.set_xlabel('Frequency (number of pixels)', fontsize=9)
    ax_reg.set_ylabel('Month' if freq == 'monthly' else'Hour (UTC)', fontsize=9)
    ax_reg.set_title('(b)', fontsize=12, loc='left')
    ax_reg.invert_yaxis()
    ax_reg.tick_params(axis='x', labelsize=7)
    ax_reg.grid(axis='x', alpha=0.3)
    ax_reg.legend(fontsize=7, loc='upper right', framealpha=0.7)

    # ── Frequência por uso do solo (usando pixel_df ponderado) ────────────────
    # Conta pixels com content > 0 que têm pixel_max == t
    # Conteúdo alinhado ao índice de gdf_metrica
    content_duna_sel = content_duna[metrica_vals.index]
    content_mine_sel = content_mine[metrica_vals.index]
    
    # GeoDataFrame base com a métrica já calculada
    gdf_uso = gdf_metrica.copy()
    gdf_uso['content_duna'] = content_duna_sel
    gdf_uso['content_mine'] = content_mine_sel
    
    # Seleciona apenas pixels onde existe cada uso do solo
    gdf_duna = gdf_uso[gdf_uso['content_duna'] > 0]
    gdf_mine = gdf_uso[gdf_uso['content_mine'] > 0]
    
    # Conta frequência de mês/hora de maior emissão
    freq_duna = np.zeros(n_periodos)
    freq_mine = np.zeros(n_periodos)
    
    for t in range(n_periodos):
        freq_duna[t] = (gdf_duna['metrica'] == t).sum()
        freq_mine[t] = (gdf_mine['metrica'] == t).sum()
    
    usos = [
        ('Dune / Sandy area' ,        freq_duna, USOS_SOLO['Dune / Sandy area']),
        ('Mining',            freq_mine, USOS_SOLO['Mining']),
    ]

    # Painel 3 – barras empilhadas por uso do solo
    bottom = np.zeros(n_periodos)
    for nome_uso, vals, cor in usos:
        ax_solo.barh(
            np.arange(n_periodos), vals,
            left=bottom,
            color=cor,
            alpha=0.55,          # alpha reduzido para duna e mineração
            edgecolor='white', linewidth=0.3,
            label=nome_uso
        )
        bottom += vals

    ax_solo.set_yticks([])
    #ax_solo.set_yticklabels(labels_per, fontsize=7)
    ax_solo.set_xlabel('Frequency (number of pixels)', fontsize=9)
    #ax_solo.set_ylabel('Mês' if freq == 'monthly' else 'Hora (UTC)', fontsize=9)
    ax_solo.set_title('(c)', fontsize=12, loc='left')
    ax_solo.invert_yaxis()
    ax_solo.tick_params(axis='x', labelsize=7)
    ax_solo.grid(axis='x', alpha=0.3)
    ax_solo.legend(fontsize=7, loc='upper right', framealpha=0.7)

    plt.tight_layout()

    if salvar:
        nome = f"{tipo}_{var}_{escala_temporal.replace('á','a')}.png"
        fig.savefig(OUTPUT_DIR + nome, dpi=150, bbox_inches='tight')
        print(f"[INFO] Figura salva em: {OUTPUT_DIR + nome}")
    if mostrar:
        plt.show()

    import pandas as pd


    if freq == 'monthly':
        indices = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    else:
        indices = [str(h) for h in range(24)]

    tabela = pd.DataFrame(index=indices)

    # Regiões
    for regiao in ['South', 'Southeast', 'Midwest', 'Northeast', 'North']:
        tabela[regiao] = freq_regiao[regiao].astype(int)

    # Uso do solo
    tabela['Dunas'] = freq_duna.astype(int)
    tabela['Mineração'] = freq_mine.astype(int)
    tabela.to_csv('/home/lcqar/BRAIN/emis/windBlowDustBR/Outputs/BR_12km/tables/obj_02_maximo_'+freq+'.csv')

    return fig


# ──────────────────────────────────────────────────────────────────────────────
# TIPO 2 – msi / amplitude / força: mapa + boxplot por região × uso do solo
# ──────────────────────────────────────────────────────────────────────────────

def plot_metrica_regiao_solo(
    dfs, lat, lon, shp,
    var, escala_temporal, tipo,
    conteudos_uso_solo=None,
    salvar=True, mostrar=False
):
    """
    Figura com 2 colunas:
      col 1 – mapa da métrica (igual a _plot_mapa)
      col 2 – boxplot horizontal:
                eixo Y = regiões
                para cada região, um boxplot por uso do solo
                eixo X = valor da métrica

    Parâmetros
    ----------
    tipo : str  'msi' | 'amplitude' | 'forca'
    conteudos_uso_solo : dict {nome: array_peso} ou None
    """
    if tipo == 'pixel_max':
        raise ValueError("Use plot_pixel_max para tipo='pixel_max'.")

    freq = 'monthly' if escala_temporal == 'mensal' else 'hourly'

    gdf_pixels = _build_gdf(lat, lon)
    pixel_df   = _pixel_dataframe(dfs, var, freq)
    metrica_vals = _calc_metrica(pixel_df, tipo, freq)
    gdf_metrica  = gdf_pixels.loc[metrica_vals.index].copy()
    gdf_metrica['metrica'] = metrica_vals.values

    if conteudos_uso_solo is None:
        conteudos_uso_solo = {nome: None for nome in USOS_SOLO}

    nomes_uso  = list(conteudos_uso_solo.keys())
    n_usos     = len(nomes_uso)
    regioes    = list(REGIOES.keys())
    n_regioes  = len(regioes)

    fig, (ax_map, ax_box) = plt.subplots(
        1, 2,
        figsize=(14, 8),
        gridspec_kw={'width_ratios': [1.2, 1], 'wspace': 0.15}
    )

    # Painel 1 – mapa
    _plot_mapa(ax_map, gdf_metrica, shp, 'Brasil', tipo, freq, metrica_vals)
    ax_map.set_title('(a)', fontsize=12,loc='left')

    # Painel 2 – boxplot por região × uso do solo
    # Posições: para cada região, n_usos boxplots lado a lado
    espacamento = n_usos + 1   # espaço entre grupos de regiões
    posicoes_centro = np.arange(n_regioes) * espacamento

    all_data = []    # lista de arrays
    all_pos  = []    # posição Y de cada boxplot
    all_cores = []   # cor de cada boxplot

    for i, regiao in enumerate(regioes):
        shp_reg = shp[shp['SIGLA_UF'].isin(REGIOES[regiao]['estados'])]
        gdf_reg = gpd.clip(gdf_metrica.dropna(subset=['metrica']), shp_reg)
        idx_reg = gdf_reg.index

        for j, nome_uso in enumerate(nomes_uso):
            peso = conteudos_uso_solo[nome_uso]
            if peso is not None:
                peso_s  = pd.Series(peso, index=gdf_pixels.index).reindex(idx_reg).fillna(0)
                vals    = gdf_reg.loc[idx_reg, 'metrica'] * peso_s
            else:
                vals    = gdf_reg.loc[idx_reg, 'metrica']
            vals = vals.dropna().values
            vals = vals[vals > 0]

            pos = posicoes_centro[i] + (j - (n_usos - 1) / 2) * 0.8
            all_data.append(vals)
            all_pos.append(pos)
            all_cores.append(USOS_SOLO.get(nome_uso, '#aaaaaa'))

    # Desenha boxplots individuais
    bp = ax_box.boxplot(
        all_data,
        positions=all_pos,
        vert=False,
        showfliers=False,
        widths=0.6,
        patch_artist=True,
        medianprops=dict(color='black', linewidth=1.5)
    )
    for patch, cor in zip(bp['boxes'], all_cores):
        patch.set_facecolor(cor)
        patch.set_alpha(0.8)

    # Ticks de região
    ax_box.set_yticks(posicoes_centro)
    ax_box.set_yticklabels(regioes, fontsize=9)
    ax_box.invert_yaxis()
    ax_box.grid(axis='x', alpha=0.3)

    label_x = {
        'amplitude': 'Amplitude (ton)',
        'forca':     'Força Relativa',
        'msi':       'MSI (Markham index)'
    }.get(tipo, 'Valor')
    ax_box.set_xlabel(label_x, fontsize=9)
    ax_box.set_ylabel('Região', fontsize=9)
    ax_box.set_title('(b)', fontsize=12, loc='left')

    # Legenda de uso do solo
    patches_leg = [
        mpatches.Patch(color=USOS_SOLO.get(n, '#aaa'), label=n)
        for n in nomes_uso
    ]
    ax_box.legend(handles=patches_leg, fontsize=8, loc='lower right', framealpha=0.7)

    plt.tight_layout()

    if salvar:
        nome = f"{tipo}_regiao_solo_{var}_{escala_temporal.replace('á','a')}.png"
        fig.savefig(OUTPUT_DIR + nome, dpi=150, bbox_inches='tight')
        print(f"[INFO] Figura salva em: {OUTPUT_DIR + nome}")
    if mostrar:
        plt.show()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# TIPO 3 – magnitude: boxplot emissão bruta × mês/hora, dividido por uso do solo
# ──────────────────────────────────────────────────────────────────────────────

def plot_magnitude(
    dfs, lat, lon, shp,
    var, escala_temporal,
    escala_espacial='Brasil',
    conteudos_uso_solo=None,
    salvar=True, mostrar=False
):
    """
    Figura com 1 coluna:
      Boxplot horizontal:
        eixo Y = mês ou hora
        eixo X = magnitude da emissão (g/s)
        para cada período, um boxplot por uso do solo

    Parâmetros
    ----------
    escala_espacial : str
        Região para filtrar pixels (padrão: 'Brasil').
    conteudos_uso_solo : dict {nome: array_peso} ou None
    """
    freq = 'monthly' if escala_temporal == 'mensal' else 'hourly'
    n_periodos = 12 if freq == 'monthly' else 24
    labels_per = LABELS_MENSAIS if freq == 'monthly' else LABELS_HORARIOS
    periodos   = list(range(n_periodos))

    gdf_pixels = _build_gdf(lat, lon)

    # Filtra pixels da região desejada
    gdf_regiao, _ = _mask_regiao(gdf_pixels, shp, escala_espacial)
    idx_validos    = gdf_regiao.index

    if conteudos_uso_solo is None:
        conteudos_uso_solo = {nome: None for nome in USOS_SOLO}

    nomes_uso = list(conteudos_uso_solo.keys())
    n_usos    = len(nomes_uso)

    # ── Coleta dados ──────────────────────────────────────────────────────────
    emissoes, _ = _emissoes_por_uso_solo(
        dfs, var, freq, idx_validos, conteudos_uso_solo
    )

    # ── Figura ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, max(6, n_periodos * 0.45)))

    espacamento = n_usos + 1
    posicoes_centro = np.arange(n_periodos) * espacamento

    all_data  = []
    all_pos   = []
    all_cores = []

    for i, t in enumerate(periodos):
        for j, nome_uso in enumerate(nomes_uso):
            vals = emissoes[nome_uso].get(t, np.array([]))
            pos  = posicoes_centro[i] + (j - (n_usos - 1) / 2) * 0.7
            all_data.append(vals if len(vals) > 0 else np.array([np.nan]))
            all_pos.append(pos)
            all_cores.append(USOS_SOLO.get(nome_uso, '#aaaaaa'))

    bp = ax.boxplot(
        all_data,
        positions=all_pos,
        vert=False,
        showfliers=False,
        widths=0.55,
        patch_artist=True,
        medianprops=dict(color='black', linewidth=1.5)
    )
    for patch, cor in zip(bp['boxes'], all_cores):
        patch.set_facecolor(cor)
        patch.set_alpha(0.8)

    ax.set_yticks(posicoes_centro)
    ax.set_yticklabels(labels_per, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Emission magnitude (g/s)', fontsize=10)
    ax.set_ylabel('Month' if freq == 'monthly' else 'Hour (UTC)', fontsize=10)
    ax.set_title(
        f'Emission magnitude – {var} | {escala_temporal} | {escala_espacial}',
        fontsize=11
    )
    ax.grid(axis='x', alpha=0.3)

    # Legenda
    patches_leg = [
        mpatches.Patch(color=USOS_SOLO.get(n, '#aaa'), label=n)
        for n in nomes_uso
    ]
    ax.legend(handles=patches_leg, fontsize=8, loc='lower right', framealpha=0.7)

    plt.tight_layout()

    if salvar:
        nome = (f"magnitude_{var}_{escala_temporal.replace('á','a')}"
                f"_{escala_espacial.replace(' ','_')}.png")
        fig.savefig(OUTPUT_DIR + nome, dpi=150, bbox_inches='tight')
        print(f"[INFO] Figura salva em: {OUTPUT_DIR + nome}")
    if mostrar:
        plt.show()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# FUNÇÃO PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────
 
def plot_analise_espacial_temporal(
    dfs,
    lat,
    lon,
    shp,
    var,
    escala_temporal,
    escala_espacial,
    tipo,
    salvar=True,
    mostrar=False
):
    """
    Gera figura combinada: mapa espacial (esq.) + boxplot (dir.).
 
    Parâmetros
    ----------
    dfs : dict
        Dicionário retornado por agg_wbd + agg_temporal (emissoes_wbd.pkl).
    lat, lon : array-like
        Coordenadas 1-D de cada pixel (ordem igual às linhas dos DataFrames).
    shp : GeoDataFrame
        Shapefile dos estados brasileiros (coluna 'SIGLA_UF').
    var : str
        Poluente: 'PM10', 'PMC' ou 'PMFINE'.
    escala_temporal : str
        'horária' ou 'mensal'.
    escala_espacial : str
        'Sul' | 'Sudeste' | 'Centro-Oeste' | 'Nordeste' | 'Norte' |
        'Brasil' | 'Solo Duna' | 'Solo Mineração'
    tipo : str
        'pixel_max'  → mês/hora com maior emissão por pixel
        'msi'        → Índice de Markham por pixel
        'amplitude'  → amplitude sazonal por pixel
        'forca'      → força relativa da sazonalidade por pixel
    salvar : bool
        Se True, salva a figura em OUTPUT_DIR.
    mostrar : bool
        Se True, chama plt.show().
 
    Retorno
    -------
    fig : matplotlib.figure.Figure
    """
 
    # ── Validações ────────────────────────────────────────────────────────────
    escalas_temporais_validas = ('horária', 'mensal')
    escalas_espaciais_validas = list(REGIOES.keys()) + ['Brasil', 'Solo Duna', 'Solo Mineração']
    tipos_validos = ('pixel_max', 'msi', 'amplitude', 'forca')
 
    if escala_temporal not in escalas_temporais_validas:
        raise ValueError(f"escala_temporal deve ser um de {escalas_temporais_validas}")
    if escala_espacial not in escalas_espaciais_validas:
        raise ValueError(f"escala_espacial deve ser um de {escalas_espaciais_validas}")
    if tipo not in tipos_validos:
        raise ValueError(f"tipo deve ser um de {tipos_validos}")
 
    freq = 'monthly' if escala_temporal == 'mensal' else 'hourly'
 
    print(f"[INFO] Iniciando análise: var={var} | tempo={escala_temporal} | "
          f"espaço={escala_espacial} | tipo={tipo}")
 
    # ── GeoDataFrame base de pixels ───────────────────────────────────────────
    gdf_pixels = _build_gdf(lat, lon)
 
    # ── Máscara da região de interesse ───────────────────────────────────────
    gdf_regiao, estados_lista = _mask_regiao(gdf_pixels, shp, escala_espacial)
 
    # ── Matriz emissões (n_pixels × n_periodos) ───────────────────────────────
    pixel_df = _pixel_dataframe(dfs, var, freq)
    print(pixel_df)
    
    # ── Ponderação por uso do solo (somente para Solo Duna / Solo Mineração) ──
    content = None   # None → sem ponderação
    if escala_espacial in SOLOS_ESPECIAIS:
        print(f"[INFO] Aplicando ponderação de uso do solo: {escala_espacial}")
        pixel_df, content = _uso_solo(pixel_df, var, escala_espacial)
        pixel_df = pixel_df * content[:, None]
 
    # Recorta pixel_df apenas para pixels da região
    idx_regiao = gdf_regiao.index
    pixel_df_reg = pixel_df.loc[pixel_df.index.isin(idx_regiao)]

    # ── Calcula métrica ───────────────────────────────────────────────────────
    metrica_vals = _calc_metrica(pixel_df_reg, tipo, freq)
 
    # GeoDataFrame da região com a métrica
    gdf_metrica = gdf_regiao.loc[metrica_vals.index]
 
    # ── Figura ────────────────────────────────────────────────────────────────
    fig, (ax_map, ax_box) = plt.subplots(
        1, 2,
        figsize=(12, 8),
        gridspec_kw={'width_ratios': [1, 0.4], 'wspace': 0.1}
    )
 
    # Painel esquerdo – mapa
    _plot_mapa(ax_map, gdf_metrica, shp, escala_espacial, tipo, freq, metrica_vals)
 
    # Painel direito – boxplot
    usa_eixo_temporal = (
        escala_espacial in ('Brasil', 'Solo Duna', 'Solo Mineração')
        or estados_lista is None
        or tipo != 'pixel_max'       # métricas resumidas por pixel → temporal faz mais sentido
    )
 
    if escala_espacial in REGIOES and tipo == 'pixel_max':
        # Boxplot por estado
        _boxplot_por_estado(ax_box, gdf_metrica, metrica_vals, shp, estados_lista, tipo, escala_espacial, content)
    else:
        # Boxplot temporal (eixo X = mês ou hora)
        _boxplot_temporal(ax_box, gdf_pixels, dfs, var, freq,
                          metrica_vals, escala_espacial, shp, tipo, content)
 
    # Título geral
    titulo_tipo = {
        'pixel_max': 'Peak emission period',
        'amplitude': 'Amplitude da sazonalidade',
        'forca':     'Força relativa da sazonalidade',
        'msi':       'MSI (Markham Index)'
    }[tipo]
    #fig.suptitle(
        #f"{var} – {titulo_tipo}\n"
        #f"{escala_espacial} | escala {escala_temporal}",
        #fontsize=13, fontweight='bold', y=1.01
    #)
 
    plt.tight_layout()
 
    # ── Salvar ────────────────────────────────────────────────────────────────
    if salvar:
        nome = (f"analise_{tipo}_{var}_{escala_temporal.replace('á','a')}"
                f"_{escala_espacial.replace(' ','_')}.png")
        caminho = OUTPUT_DIR + nome
        fig.savefig(caminho, dpi=150, bbox_inches='tight')
        print(f"[INFO] Figura salva em: {caminho}")
 
    if mostrar:
        plt.show()
 
    return fig
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Função auxiliar: calcula e retorna DataFrame de métricas por pixel
# ──────────────────────────────────────────────────────────────────────────────
 
def calcular_metricas_pixels(dfs, var, freq_key='monthly'):
    """
    Calcula todas as métricas por pixel para um poluente e frequência.
 
    Parâmetros
    ----------
    dfs : dict
    var : str  ('PM10', 'PMC', 'PMFINE')
    freq_key : str  ('monthly' | 'hourly')
 
    Retorno
    -------
    DataFrame com colunas: pixel_max, amplitude, forca, msi
    """
    pixel_df = _pixel_dataframe(dfs, var, freq_key)
    freq_label = 'mensal' if freq_key == 'monthly' else 'horária'
 
    resultado = pd.DataFrame(index=pixel_df.index)
    for tipo in ('pixel_max', 'amplitude', 'forca', 'msi'):
        resultado[tipo] = _calc_metrica(pixel_df, tipo, freq_label)
 
    return resultado

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
 
# ──────────────────────────────────────────────────────────────────────────────
# Bloco de execução de exemplo
# ──────────────────────────────────────────────────────────────────────────────

def extrair_ciclo_sazonal(dfs, var, freq):
    """
    Extrai as médias mensais/horárias ao longo dos anos e a variância total.
    Implementa a base para a lógica corrigida de MSI, Amplitude e Força Relativa.
    """
    periodos = sorted(dfs[var][freq].keys())
    means_dict = {}
    all_raw_arrays = []
    
    for t in periodos:
        df_t = dfs[var][freq][t]
        # Correção: tira a média ao longo das ocorrências (ex: todos os janeiros) 
        means_dict[t] = df_t.mean(axis=1).values
        # Acumula dados brutos para o cálculo correto da variância total da série
        all_raw_arrays.append(df_t.values)
        
    df_means = pd.DataFrame(means_dict, index=dfs[var][freq][periodos[0]].index)
    
    # Variância total da série histórica completa (ddof=1 igual ao .var() do pandas)
    raw_matrix = np.column_stack(all_raw_arrays)
    var_total = np.nanvar(raw_matrix, axis=1, ddof=1)
    
    return df_means, var_total

def _calc_metrica_corrigida(df_means, var_total, tipo, freq_label):
    """
    Calcula as métricas vetorizadas reproduzindo fielmente a função seasonality_analysis.
    """
    vals = df_means.values
    n_periodos = vals.shape[1]
    
    if tipo == 'amplitude':
        # amplitude (máx - mín) entre as médias
        amp = np.nanmax(vals, axis=1) - np.nanmin(vals, axis=1)
        return pd.Series(amp, index=df_means.index)*3600*1e-6
        
    elif tipo == 'forca':
        # força relativa = variância sazonal / variância total
        var_seasonal = np.nanvar(vals, axis=1, ddof=1)
        forca = np.where(var_total > 0, var_seasonal / var_total, np.nan)
        return pd.Series(forca, index=df_means.index)
        
    elif tipo == 'msi':
        # MSI (Markham Seasonality Index)
        total = np.nansum(vals, axis=1, keepdims=True)
        total = np.where(total == 0, np.nan, total)
        rel_freq = vals / total
        arr = (np.abs(rel_freq - 1/n_periodos)) / (2 * (1 - 1/n_periodos))
        msi = np.where(
            np.isnan(arr).all(axis=1),
            np.nan,
            np.nansum(arr, axis=1)
        )
        return pd.Series(msi, index=df_means.index)
    
    else:
        raise ValueError("Use apenas 'msi', 'amplitude' ou 'forca' nesta função.")


# ──────────────────────────────────────────────────────────────────────────────
# TIPO CORRIGIDO – Boxplots Sazonais Pareados (Conforme Imagem/Rascunho)
# ──────────────────────────────────────────────────────────────────────────────

def plot_sazonalidade_boxplots(
    dfs, lat, lon, shp,
    var, escala_temporal, tipo,
    salvar=True, mostrar=False
):
    """
    Figura com 2 painéis:
      Esq – Mapa com a métrica espacial (Brasil inteiro).
      Dir – Boxplot pareado por região (Eixo Y: S, SE, CO, NE, N) 
            dividido entre Mineração (Cinza) e Dunas (Laranja).
    """
    if tipo == 'pixel_max':
        raise ValueError("Esta função não se aplica a 'pixel_max'. Use plot_pixel_max.")
        
    freq = 'monthly' if escala_temporal == 'mensal' else 'hourly'
    
    # 1. Preparação dos dados e geometria
    gdf_pixels = _build_gdf(lat, lon)
    df_means, var_total = extrair_ciclo_sazonal(dfs, var, freq)
    
    # 2. Obtenção das máscaras de uso do solo
    # Usamos o _uso_solo apenas para extrair os arrays `content` de dunas e mineração
    _, content_duna = _uso_solo(df_means, var, 'Solo Duna')
    _, content_mine = _uso_solo(df_means, var, 'Solo Mineração')
    
    # 3. Cálculo da métrica corrigida e alocação no GeoDataFrame
    metrica_vals = _calc_metrica_corrigida(df_means, var_total, tipo, freq)
    
    gdf_metrica = gdf_pixels.loc[metrica_vals.index].copy()
    gdf_metrica['metrica'] = metrica_vals.values
    gdf_metrica['content_duna'] = content_duna
    gdf_metrica['content_mine'] = content_mine

    # 4. Construção da Figura
    fig, axes = plt.subplots(
        1, 2,
        figsize=(10, 6),
        gridspec_kw={'width_ratios': [1.2, 0.5], 'wspace': -0.05}
    )
    ax_map, ax_box = axes
    
    # Painel 1: Mapa (reaproveitando a lógica original)
    _plot_mapa(ax_map, gdf_metrica, shp, 'Brasil', tipo, freq, metrica_vals)
    ax_map.set_title('(a)', fontsize=12, loc='left')
    # Painel 2: Boxplots Pareados por Região
    regioes_ordem = ['North', 'Northeast', 'Midwest', 'Southeast', 'South']
    siglas_map = {'South': 'S', 'Southeast': 'SE', 'Midwest': 'CO', 'Northeast': 'NE', 'North': 'N'}
    
    dados_duna = []
    dados_mine = []
    y_positions = np.arange(len(regioes_ordem))
    
    # Extrai os dados validos filtrando quem possui fração do solo > 0
    # Diferente do plot anterior, não multiplicamos a métrica pelo content para não distorcer os índices.
    for regiao in regioes_ordem:
        estados = REGIOES[regiao]['estados']
        shp_reg = shp[shp['SIGLA_UF'].isin(estados)]
        
        gdf_reg = gpd.sjoin(
            gdf_metrica,
            shp_reg[['geometry']],
            predicate='intersects',
            how='inner'
        ).drop(columns='index_right')
        
        vals_duna = gdf_reg[gdf_reg['content_duna'] > 0]['metrica'].values
        vals_mine = gdf_reg[gdf_reg['content_mine'] > 0]['metrica'].values
        
        vals_duna = vals_duna[~np.isnan(vals_duna)]
        vals_mine = vals_mine[~np.isnan(vals_mine)]

        dados_duna.append(vals_duna if len(vals_duna) > 0 else np.array([np.nan]))
        dados_mine.append(vals_mine if len(vals_mine) > 0 else np.array([np.nan]))

    # Configurações visuais que replicam o rascunho
    box_width = 0.28
    offset = 0.17
    cor_duna = '#D2A679'
    cor_mine = '#808080'
    
    flierprops = dict(marker='o', markersize=2, linestyle='none')
    
    # Boxplot Mineração (Deslocado para cima)
    bp_mine = ax_box.boxplot(
        dados_mine,
        positions=y_positions + offset, flierprops=flierprops,
        vert=False, widths=box_width, patch_artist=True, showfliers=True,
        boxprops=dict(facecolor='white', color=cor_mine, linewidth=1.5),
        whiskerprops=dict(color=cor_mine, linewidth=1.5),
        capprops=dict(color=cor_mine, linewidth=1.5),
        medianprops=dict(color=cor_mine, linewidth=1.5)
    )
    
    # Boxplot Dunas (Deslocado para baixo)
    bp_duna = ax_box.boxplot(
        dados_duna,
        positions=y_positions - offset,flierprops=flierprops,
        vert=False, widths=box_width, patch_artist=True, showfliers=True,
        boxprops=dict(facecolor='white', color=cor_duna, linewidth=1.5),
        whiskerprops=dict(color=cor_duna, linewidth=1.5),
        capprops=dict(color=cor_duna, linewidth=1.5),
        medianprops=dict(color=cor_duna, linewidth=1.5)
    )
    
    # Ajustes do Eixo Y (Posicionamento e Siglas)
    ax_box.set_yticks(y_positions)
    ax_box.set_yticklabels([siglas_map[r] for r in regioes_ordem], fontsize=10)
    
    # Inverte o eixo Y para o "Sul" (S) ficar no topo, conforme seu desenho
    ax_box.invert_yaxis()
    
    # Customização do Grid e Rótulos
    ax_box.grid(axis='x', alpha=0.3, linestyle='--')
    
    label_x = {
        'amplitude': 'Amplitude (ton)',
        'forca':     'Força Relativa',
        'msi':       'MSI (Markham index)'
    }.get(tipo, tipo.upper())
    ax_box.set_xlabel(label_x, fontsize=10)
    
    # Legenda customizada
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='white', edgecolor=cor_mine, label='Mining', linewidth=1.5),
        Patch(facecolor='white', edgecolor=cor_duna, label='Dune / Sandy area', linewidth=1.5)
    ]
    ax_box.legend(handles=legend_elements, loc='upper left', framealpha=0.8, fontsize=9)
    
    if tipo == 'msi':
        ax_box.set_xlim(-0.05, 1.05)
    else:
        xmax = ax_box.get_xlim()[1]
        xmin = ax_box.get_xlim()[0]
        if tipo == 'forca':
            ax_box.set_xlim(0, xmax)
        elif tipo == 'amplitude':
            print(xmin)
            ax_box.set_xlim(0.0000001, xmax)
            ax_box.set_xscale('log')

    ax_box.set_title('(b)', fontsize=12, loc='left')
    plt.tight_layout()
    
    if salvar:
        nome = f"boxplots_{tipo}_{var}_{escala_temporal.replace('á','a')}.png"
        fig.savefig(OUTPUT_DIR + nome, dpi=150, bbox_inches='tight')
        print(f"[INFO] Figura salva em: {OUTPUT_DIR + nome}")
        
    if mostrar:
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

    tabela.to_csv('/home/lcqar/BRAIN/emis/windBlowDustBR/Outputs/BR_12km/tables/obj_02_msi_'+freq+'.csv')
    
    return fig

def plot_pixel_sum(
    dfs, lat, lon, shp,
    var, escala_temporal,
    salvar=True, mostrar=False
):
    """
    Figura com 3 colunas, análoga a plot_pixel_max, mas com métrica de soma total:
      col 1 – mapa com a soma total de emissão de cada pixel (todos os períodos)
      col 2 – barras empilhadas: soma de emissão por mês/hora (eixo X),
               eixo Y = meses/horas, cores = regiões
      col 3 – barras empilhadas: mesma soma por uso do solo (duna e mineração),
               usando pixel_df_duna e pixel_df_mine já ponderados
    """
    freq = 'monthly' if escala_temporal == 'mensal' else 'hourly'
    n_periodos = 12 if freq == 'monthly' else 24
    labels_per = LABELS_MENSAIS if freq == 'monthly' else LABELS_HORARIOS

    # ── GDF completo do Brasil ────────────────────────────────────────────────
    gdf_pixels = _build_gdf(lat, lon)
    pixel_df   = _pixel_dataframe(dfs, var, freq)   # linhas = pixels, colunas = períodos

    # ── Ponderação por uso do solo ────────────────────────────────────────────
    pixel_df_d, content_duna = _uso_solo(pixel_df, var, 'Solo Duna')
    pixel_df_duna = pixel_df_d * content_duna[:, None]*3600*1e-6
    print(1)
    pixel_df_m, content_mine = _uso_solo(pixel_df, var, 'Solo Mineração')
    pixel_df_mine = pixel_df_m * content_mine[:, None]*3600*1e-6
    print(2)
    # ── Métrica = soma total por pixel (todos os períodos), para o mapa ───────
    pixel_df_base = _pixel_dataframe(dfs, var, freq)
    metrica_vals  = pixel_df_base.sum(axis=1)*3600*1e-6     # soma ao longo das colunas (períodos)
    metrica_vals  = metrica_vals.dropna()
    print(3)
    gdf_metrica = gdf_pixels.loc[metrica_vals.index].copy()
    gdf_metrica['metrica'] = metrica_vals.values

    # ── Figura ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        1, 3,
        figsize=(10, 5),
        gridspec_kw={'width_ratios': [1.4, 0.6, 0.6], 'wspace': 0.15}
    )
    ax_map, ax_reg, ax_solo = axes

    # Painel 1 – mapa (escala contínua de cores, não categórica por períodos
    _plot_mapa(ax_map, gdf_metrica, shp, 'Brasil', 'Magnitude', freq, metrica_vals)
    ax_map.set_title('(a)', fontsize=12,loc='left')

    # ── Soma por região (valores brutos por período, não ponderados) ─────────
    soma_regiao = {r: np.zeros(n_periodos) for r in REGIOES}
    for regiao, info in REGIOES.items():
        shp_reg = shp[shp['SIGLA_UF'].isin(info['estados'])]
        gdf_reg = gpd.sjoin(
            gdf_metrica,
            shp_reg[['geometry']],
            predicate='intersects',
            how='inner'
        ).drop(columns='index_right')
        bloco_reg = pixel_df_base.loc[gdf_reg.index]
        for t in range(n_periodos):
            soma_regiao[regiao][t] = bloco_reg.iloc[:, t].sum()*3600*1e-6

    # Painel 2 – barras empilhadas por região (soma, não frequência)
    bottom = np.zeros(n_periodos)
    for regiao, info in REGIOES.items():
        vals = soma_regiao[regiao]
        ax_reg.barh(
            np.arange(n_periodos), vals,
            left=bottom,
            color=info['cor'],
            alpha=0.55,
            edgecolor='white', linewidth=0.3,
            label=regiao
        )
        bottom += vals

    ax_reg.set_yticks(np.arange(n_periodos))
    ax_reg.set_yticklabels(labels_per, fontsize=7)
    ax_reg.set_xlabel('PM$_{{10}}$ emission (ton)', fontsize=9)
    ax_reg.set_ylabel('Month' if freq == 'monthly' else 'Hour (UTC)', fontsize=9)
    ax_reg.set_title('(b)', fontsize=12, loc='left')
    ax_reg.invert_yaxis()
    ax_reg.tick_params(axis='x', labelsize=7)
    ax_reg.grid(axis='x', alpha=0.3)
    ax_reg.legend(fontsize=7, loc='upper right', framealpha=0.7)

    # ── Soma por uso do solo (já ponderada por content_duna / content_mine) ──
    soma_duna = np.asarray(pixel_df_duna.sum(axis=0))   # soma por período, todos os pixels
    soma_mine = np.asarray(pixel_df_mine.sum(axis=0))

    usos = [
        ('Dune / Sandy area',         soma_duna, USOS_SOLO['Dune / Sandy area']),
        ('Mining',            soma_mine, USOS_SOLO['Mining']),
    ]

    # Painel 3 – barras empilhadas por uso do solo (soma)
    bottom = np.zeros(n_periodos)
    for nome_uso, vals, cor in usos:
        ax_solo.barh(
            np.arange(n_periodos), vals,
            left=bottom,
            color=cor,
            alpha=0.55,
            edgecolor='white', linewidth=0.3,
            label=nome_uso
        )
        bottom += vals


    ax_solo.set_yticks([])
    #ax_solo.set_yticklabels(labels_per, fontsize=7)
    ax_solo.set_xlabel('PM$_{{10}}$ emission (ton)', fontsize=9)
    #ax_solo.set_ylabel('Mês' if freq == 'monthly' else 'Hora (UTC)', fontsize=9)
    ax_solo.set_title('(c)', fontsize=12, loc='left')
    ax_solo.invert_yaxis()
    ax_solo.tick_params(axis='x', labelsize=7)
    ax_solo.grid(axis='x', alpha=0.3)
    ax_solo.legend(fontsize=7, loc='upper right', framealpha=0.7)

    plt.tight_layout()

    if salvar:
        nome = f"soma_{var}_{escala_temporal.replace('á','a')}.png"
        fig.savefig(OUTPUT_DIR + nome, dpi=150, bbox_inches='tight')
        print(f"[INFO] Figura salva em: {OUTPUT_DIR + nome}")
    if mostrar:
        plt.show()

    if freq == 'monthly':
        indices = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    else:
        indices = [str(h) for h in range(24)]

    tabela = pd.DataFrame(index=indices)

    # Regiões
    for regiao in ['South', 'Southeast', 'Midwest', 'Northeast', 'North']:
        tabela[regiao] = soma_regiao[regiao].astype(float)

    # Uso do solo
    tabela['Dunas'] = soma_duna.astype(float)
    tabela['Mineração'] = soma_mine.astype(float)
    tabela.to_csv('/home/lcqar/BRAIN/emis/windBlowDustBR/Outputs/BR_12km/tables/obj_02_magnitude_'+freq+'.csv')

    return fig

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

if __name__ == "__main__":
 
    import sys
    import os
 
    # Adiciona o diretório do objetivo02.py ao path para importar funções auxiliares
    sys.path.insert(0, os.path.dirname(__file__))
 
    # ── Carrega dados ────────────────────────────────────────────────────────
    print("[INFO] Carregando pickle...")
    with open(PKL_PATH, 'rb') as f:
        dict_dfs = pickle.load(f)
 
    print("[INFO] Lendo coordenadas...")
    nc_exemplo = (
        "/home/lcqar/BRAIN/emis/windBlowDustBR/Outputs/BR_12km/"
        "windBlowDust_PM10_2023-02-24-00:00:00_2023-02-25-00:00:00.nc"
    )
    lon, lat = latlon_2d(nc_exemplo)
 
    print("[INFO] Lendo shapefile...")
    shp = gpd.read_file(SHP_PATH, engine="pyogrio")
    

    # ── Prepara pesos de uso do solo (substitua pelos seus arrays reais) ──────
    # Cada entrada é um array de tamanho n_pixels com fração de cobertura [0,1].
    # Enquanto não há máscaras reais, use None (sem ponderação / peso uniforme).
    conteudos_uso_solo = {
        'Dune / Sandy area':         None,   # substituir por array real de peso
        'Mining':            None,   # substituir por array real de peso
    }

    # ── Exemplos de uso ──────────────────────────────────────────────────────
    
    # TIPO 1 – pixel_max: mapa + barras empilhadas (regiões e uso do solo)
    '''for tipo in ['pixel_max']:
        for temp in ['mensal', 'horária']:
            plot_pixel_max(
                dfs=dict_dfs, tipo=tipo, lat=lat, lon=lon, shp=shp,
                var='PM10',
                escala_temporal=temp,
                salvar=True, mostrar=False
            )
    
    # TIPO 2 – msi / amplitude / força: mapa + boxplot região × uso do solo
    for tipo in ['msi']:
        for temp in ['mensal', 'horária']:
            plot_sazonalidade_boxplots(
                dfs=dict_dfs, lat=lat, lon=lon, shp=shp,
                var='PM10', escala_temporal=temp, tipo=tipo,
                salvar=True, mostrar=False
            )
    '''
    # TIPO 3 – magnitude: boxplot emissão bruta × mês/hora por uso do solo
    for temp in ['mensal', 'horária']:
        plot_pixel_sum(
            dfs=dict_dfs, lat=lat, lon=lon, shp=shp,
            var='PM10', escala_temporal=temp,
            salvar=True, mostrar=False
        )
    
    # ── Funções originais (mantidas) ─────────────────────────────────────────
    """for tip in ['pixel_max', 'amplitude', 'forca', 'msi']:
        for temp in ['mensal', 'horária']:
            for espa in ['Solo Duna', 'Solo Mineração', 'Nordeste', 'Sul',
                         'Centro-Oeste', 'Sudeste', 'Norte', 'Brasil']:
                plot_analise_espacial_temporal(
                    dfs=dict_dfs, lat=lat, lon=lon, shp=shp,
                    var='PM10',
                    escala_temporal=temp,
                    escala_espacial=espa,
                    tipo=tip,
                    salvar=True, mostrar=False
                )
    """
    print("[INFO] Concluído.")
