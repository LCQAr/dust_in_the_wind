#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-----------------scatter_windBlowDust_2023.py----------------------------------

Script para gerar scatter plot de sensibilidade do modelo de ressuspensão eólica.

Eixo X : velocidade do vento (u10, m/s) — flatten timestep × pixel
Eixo Y : fluxo vertical normalizado pela área (Fvtot / area_total, g s⁻¹ m⁻²)
Cor    : quartil de umidade do solo (SMOIS) — cores discretas:
           P0–25   → vermelho (#d62728)
           P25–50  → amarelo  (#f5c518)
           P50–75  → verde    (#2ca02c)
           P75–100 → azul     (#1f77b4)

Gera duas figuras:
  1. scatter_sensitivity_YYYY.png      — scatter principal com cores por quartil
  2. scatter_sensitivity_YYYY_kde.png  — subplot 2×2 com KDE gaussiana por quartil

Lê os outputs já gerados pelo shRunnerWindBlowDust.py para todos os dias de 2023.
Os dados meteorológicos (vento, umidade) são lidos diretamente dos arquivos WRF.
O fluxo vertical PM10 é lido dos netCDFs de output do modelo.
A área total dos pixels é lida dos arquivos intermediários gerados pelo regMap.

Uso:
    python scatter_windBlowDust_2023.py \
        --windBlowDustFolder /home/lcqar/BRAIN/emis/windBlowDustBR \
        --wrfoutFolder /caminho/para/wrfout \
        --domain 2 \
        --GDNAM BR_12km \
        --outfig /home/lcqar/BRAIN/emis/windBlowDustBR/Outputs/BR_12km/figures/scatter_2023.png

@author: adaptado de leohoinaski / gerado com assistência Claude
"""

import argparse
import os
import sys
import glob
import numpy as np
import netCDF4 as nc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from datetime import datetime, timedelta
from pathlib import Path
import resource
import gridDetails as grd
import regridMAPBIOMAS as regMap
from matplotlib.colors import LogNorm
from scipy.stats import gaussian_kde

RAM_LIMIT_GB = 30

soft, hard = resource.getrlimit(resource.RLIMIT_AS)
resource.setrlimit(
    resource.RLIMIT_AS,
    (RAM_LIMIT_GB * 1024**3, hard)
)

# =============================================================================
# Funções auxiliares (replicadas de metPrep.py para não depender do módulo)
# =============================================================================

def ustarCalc(uz, z0):
    """Calcula friction velocity u* a partir de u10 e z0."""
    k = 0.4
    z = 10.0
    z0 = np.array(z0, dtype=float, copy=True)
    z0[z0 <= 0] = np.nan
    ustar = k * uz / np.log(z / z0)
    return ustar


def roughness(avWRF):
    """Calcula rugosidade z0 a partir da fração de área vegetada."""
    hs = 0.02
    hv = 0.1
    av = np.array(avWRF, copy=True, dtype=float)
    av[av > 0.046] = 0.046
    lambdaS = 0.002
    lambdaV = -0.35 * np.log(1 - av)
    lambdaSum = lambdaS + lambdaV
    h = (hv * lambdaV + hs * lambdaS) / (lambdaSum)
    z0 = np.array(lambdaSum, copy=True)
    z0[lambdaSum >= 0.2] = ((0.083 * lambdaSum ** (-0.46)) * h)[lambdaSum >= 0.2]
    z0[lambdaSum < 0.2]  = ((0.96  * lambdaSum ** (1.07))  * h)[lambdaSum < 0.2]
    return z0


# =============================================================================
# Leitura do netCDF de emissão PM10
# =============================================================================

def load_pm10_flux(nc_path):
    """
    Lê o fluxo vertical PM10 do netCDF de output.
    Retorna array (tempo, lat, lon) em g/s.
    Tenta variáveis comuns: 'PM10', 'Fv', 'FV', 'EMIS', 'VAR_1'.
    """
    ds = nc.Dataset(nc_path, 'r')
    candidates = ['PM10', 'Fv', 'FV', 'EMIS', 'VAR_1', 'windBlowDust', 'PMTOT']
    var = None
    for c in candidates:
        if c in ds.variables:
            var = c
            break
    if var is None:
        # Pega a primeira variável não dimensional com 3+ dimensões
        for v in ds.variables:
            if ds.variables[v].ndim >= 3 and v not in ('lat','lon','latitude','longitude','time','Times'):
                var = v
                break
    if var is None:
        ds.close()
        raise ValueError(f"Nenhuma variável de emissão encontrada em {nc_path}. "
                         f"Variáveis disponíveis: {list(ds.variables.keys())}")
    data = np.array(ds.variables[var][:])
    ds.close()
    # Garante shape (tempo, lat, lon) — remove dimensões extras (camadas, etc.)
    while data.ndim > 3:
        data = data[:, 0, :, :]   # pega primeira camada vertical
    return data  # g/s


# =============================================================================
# Leitura da área total dos pixels
# =============================================================================

def load_area_total(outfolder, GDNAM):
    """
    Tenta carregar area_total de um arquivo .npy salvo pelo regMap,
    ou infere a partir de um netCDF de emissão já existente se não encontrar.
    Retorna array (lat, lon) em m².
    """
    inputFolder = 'home/lcqar/BRAIN/emis/windBlowDustBR/inputs'
    idSoils = [23,30]
    mcipMETCRO3Dpath = '/home/lcqar/CMAQ_REPOv5.4/data/mcip/BR_12km/METCRO3D_'+GDNAM+'_2023-01-01.nc'
    mcipGRIDDOT2Dpath = '/home/lcqar/CMAQ_REPOv5.4/data/mcip/BR_12km/GRIDDOT2D_'+GDNAM+'_2023-01-01.nc'

    ds,datesTime,lia,domainShp,lat,lon,lat_index,lon_index,grids = grd.main(
                                          mcipMETCRO3Dpath,mcipGRIDDOT2Dpath,'/home/lcqar/GAR_BR/WRF/2023/2023_01/','d02')

    av,al,alarea,lat,lon,domainShp = regMap.main(GDNAM,inputFolder,
                                                     outfolder,'2023',idSoils,False,
                                                     grids,domainShp,lat,lon)
    
    area_total = np.nansum(alarea,axis=0)

    return area_total
    
    '''
    # Opção 1: arquivo .npy salvo pelo pipeline
    npy_candidates = [
        os.path.join(outfolder, 'area_total.npy'),
        os.path.join(outfolder, f'area_total_{GDNAM}.npy'),
        os.path.join(outfolder, 'alarea.npy'),
    ]
    for f in npy_candidates:
        if os.path.exists(f):
            arr = np.load(f)
            # alarea pode ter shape (soilId, lat, lon) — soma sobre soils
            if arr.ndim == 3:
                arr = np.nansum(arr, axis=0)
            print(f"  area_total carregada de {f}  shape={arr.shape}")
            return arr

    # Opção 2: arquivo netCDF com variável de área
    nc_area_candidates = glob.glob(os.path.join(outfolder, '*area*'))
    for f in nc_area_candidates:
        if f.endswith('.nc'):
            try:
                ds = nc.Dataset(f, 'r')
                if 'area' in ds.variables:
                    arr = np.array(ds.variables['area'][:])
                    ds.close()
                    if arr.ndim == 3:
                        arr = np.nansum(arr, axis=0)
                    print(f"  area_total carregada de {f}")
                    return arr
                ds.close()
            except Exception:
                pass
    '''
    # Fallback: retorna None (o scatter usará Fvtot sem normalização e avisará)
    print("  AVISO: area_total não encontrada — usando Fvtot sem normalização pela área.")
    return None


# =============================================================================
# Leitura dos dados WRF (vento u10 e umidade SMOIS)
# =============================================================================

def load_wrf_met(wrfout_path, lat_index, lon_index, lia):
    """
    Extrai u10 e SMOIS do wrfout.
    Retorna uz (tempo, lat, lon) e w (tempo, lat, lon).
    """
    try:
        import wrf
        ds = nc.Dataset(wrfout_path, 'r')
        uz_full = np.array(
            wrf.g_wind.get_destag_wspd_wdir10(ds, timeidx=wrf.ALL_TIMES)[0,
                                                                          lia,
                                                                          lat_index,
                                                                          lon_index]
        )
        uz = uz_full[:, :-1, :-1]
        w_full = ds['SMOIS'][lia, 0, lat_index, lon_index]
        w = np.array(w_full)[:, :-1, :-1]
        avWRF_full = ds['VEGFRA'][lia, lat_index, lon_index] / 100
        avWRF = np.array(avWRF_full)[:, :-1, :-1]
        ds.close()
        return uz, w, avWRF
    except Exception as e:
        print(f"  ERRO ao ler WRF com wrf-python: {e}")
        print("  Tentando leitura direta sem wrf-python...")
        ds = nc.Dataset(wrfout_path, 'r')
        # u10 como magnitude de U10 e V10
        u10 = np.array(ds.variables['U10'][lia, lat_index, lon_index])[:, :-1, :-1]
        v10 = np.array(ds.variables['V10'][lia, lat_index, lon_index])[:, :-1, :-1]
        uz = np.sqrt(u10**2 + v10**2)
        w = np.array(ds.variables['SMOIS'][lia, 0, lat_index, lon_index])[:, :-1, :-1]
        avWRF = np.array(ds.variables['VEGFRA'][lia, lat_index, lon_index])[:, :-1, :-1] / 100
        ds.close()
        return uz, w, avWRF


# =============================================================================
# Descoberta dos arquivos de output PM10
# =============================================================================

def find_pm10_files(outfolder, year=2023):
    """
    Encontra todos os netCDFs de PM10 para o ano especificado.
    Padrão esperado: windBlowDust_PM10_YYYY-MM-DD-00:00:00_YYYY-MM-DD-00:00:00
    """
    pattern = os.path.join(outfolder, f'windBlowDust_PM10_{year}-*')
    files = sorted(glob.glob(pattern))
    if not files:
        # Tenta sem separador de hora
        pattern2 = os.path.join(outfolder, f'windBlowDust_PM10_{year}*')
        files = sorted(glob.glob(pattern2))
    print(f"  {len(files)} arquivo(s) PM10 encontrado(s) para {year}")
    return files


# =============================================================================
# Descoberta dos arquivos WRF
# =============================================================================

def find_wrf_files(wrfoutFolder, domain, year=2023):
    """
    Encontra os wrfout para o ano especificado.
    Padrão: wrfout_<domain>_YYYY-MM-DD_*
    """

    dom_str = f'd0{domain}' if str(domain).isdigit() else domain

    files = sorted(
        Path(wrfoutFolder).rglob(
            f"wrfout_{dom_str}_{year}-*"
        )
    )

    files = [str(f) for f in files]

    print(f"  {len(files)} arquivo(s) WRF encontrado(s) para {year}")

    return files


# =============================================================================
# Extração dos índices lat/lon do netCDF de emissão
# =============================================================================

def get_grid_indices(nc_path):
    """
    Extrai dimensões de lat e lon do netCDF para gerar os índices de grade.
    Retorna lat_index, lon_index como slices completos.
    """
    ds = nc.Dataset(nc_path, 'r')
    # Tenta pegar shape da primeira variável 3D
    for v in ds.variables:
        arr = ds.variables[v]
        if arr.ndim >= 3:
            shape = arr.shape
            ds.close()
            # shape: (time, lat, lon) ou (time, layer, lat, lon)
            if len(shape) == 3:
                nlat, nlon = shape[1], shape[2]
            else:
                nlat, nlon = shape[2], shape[3]
            return slice(0, nlat + 1), slice(0, nlon + 1)
    ds.close()
    return slice(None), slice(None)


# =============================================================================
# Loop principal de coleta de dados
# =============================================================================

def collect_annual_data(pm10_files, wrfout_files, outfolder, GDNAM):
    """
    Para cada dia disponível:
      - Lê Fvtot do netCDF PM10
      - Lê uz e w do WRF correspondente
      - Calcula z0 → ustar
      - Normaliza Fvtot pela area_total
      - Faz flatten de tudo
    Retorna arrays 1D: uz_all, w_all, fv_norm_all
    """
    area_total = load_area_total(outfolder, GDNAM)

    uz_all  = []
    w_all   = []
    fv_all  = []

    # Mapeia datas dos arquivos WRF: {YYYY-MM-DD: path}
    wrf_map = {}
    for f in wrfout_files:
        basename = os.path.basename(f)
        # wrfout_d02_2023-04-01_00:00:00
        parts = basename.split('_')
        if len(parts) >= 3:
            date_str = parts[2]  # YYYY-MM-DD
            wrf_map[date_str] = f

    for pm10_path in pm10_files:
        basename = os.path.basename(pm10_path)
        # windBlowDust_PM10_2023-04-01-00:00:00_2023-04-01-00:00:00
        parts = basename.split('_')
        # Extrai YYYY-MM-DD da parte da data
        date_str = None
        for p in parts:
            if len(p) >= 10 and p[:4].isdigit() and p[4] == '-':
                date_str = p[:10]
                break
        if date_str is None:
            print(f"  Não foi possível extrair data de {basename}, pulando.")
            continue

        # Verifica se existe WRF para esse dia
        if date_str not in wrf_map:
            print(f"  WRF não encontrado para {date_str}, pulando.")
            continue

        wrfout_path = wrf_map[date_str]
        print(f"  Processando {date_str}...")

        try:
            # Lê fluxo PM10 — shape (tempo, lat, lon)
            fv = load_pm10_flux(pm10_path)   # g/s
            nlat_fv, nlon_fv = fv.shape[1], fv.shape[2]

            # Índices de grade compatíveis com metPrep.py
            lat_index = slice(0, nlat_fv + 1)
            lon_index = slice(0, nlon_fv + 1)
            lia = slice(0, fv.shape[0])       # todos os timesteps do dia

            # Lê meteorologia
            uz, w, avWRF = load_wrf_met(wrfout_path, lat_index, lon_index, lia)

            # Garante shapes compatíveis
            min_t = min(fv.shape[0], uz.shape[0], w.shape[0])
            fv   = fv[:min_t, :, :]
            uz   = uz[:min_t, :, :]
            w    = w[:min_t,  :, :]
            avWRF = avWRF[:min_t, :, :]

            # Calcula z0 e ustar (média temporal de avWRF)
            avWRF_mean = np.nanmean(avWRF, axis=0)
            z0 = roughness(avWRF_mean)
            # Replica z0 para todos os timesteps
            z0_3d = np.broadcast_to(z0[np.newaxis, :, :], uz.shape).copy()
            ustar = ustarCalc(uz, z0_3d)

            # Normaliza pela área
            if area_total is not None:
                # Garante shape compatível (pode ter 1 soilId extra)
                area_2d = area_total
                print(area_2d)
                if area_2d.shape[0] > fv.shape[1]:
                    area_2d = area_2d[:fv.shape[1], :fv.shape[2]]
                if area_2d.shape[1] > fv.shape[2]:
                    area_2d = area_2d[:fv.shape[1], :fv.shape[2]]
                # Evita divisão por zero
                area_2d = np.where(area_2d > 0, area_2d, np.nan)
                fv_norm = fv / area_2d[np.newaxis, :, :]  # g s⁻¹ m⁻²
            else:
                fv_norm = fv  # sem normalização

            # Flatten — um ponto por (timestep × pixel)
            # Remove pontos onde Fv == 0 (sem emissão) e NaN
            mask = (fv_norm > 0) & np.isfinite(uz) & np.isfinite(w) & np.isfinite(fv_norm)

            uz_all.append(uz[mask].flatten())
            w_all.append(w[mask].flatten())
            fv_all.append(fv_norm[mask].flatten())

        except Exception as e:
            print(f"  ERRO ao processar {date_str}: {e}")
            continue

    if not uz_all:
        raise RuntimeError("Nenhum dado válido encontrado. Verifique os caminhos.")

    uz_all = np.concatenate(uz_all)
    w_all  = np.concatenate(w_all)
    fv_all = np.concatenate(fv_all)

    print(f"\n  Total de pontos válidos: {len(uz_all):,}")
    print(f"  uz    range: [{uz_all.min():.3f}, {uz_all.max():.3f}] m/s")
    print(f"  w     range: [{w_all.min():.4f}, {w_all.max():.4f}] m³/m³")
    print(f"  Fvnorm range: [{fv_all.min():.2e}, {fv_all.max():.2e}]")

    return uz_all, w_all, fv_all


# =============================================================================
# Plot
# =============================================================================

def make_scatter_2(uz_all, w_all, fv_all, outfig, area_normalized=True):
    """
    Gera o scatter plot:
      X = vento u10 (m/s)
      Y = Fvtot normalizado (g s⁻¹ m⁻²) — escala log
      Cor = umidade do solo (vermelho=mín, azul=máx)
    """
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)

    # Colormap: vermelho (baixa umidade) → azul (alta umidade)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'dry_wet', ['#d62728', '#ff7f0e', '#1f77b4'], N=256
    )

    # Normalização da cor pela umidade
    w_min, w_max = np.nanpercentile(w_all, 2), np.nanpercentile(w_all, 98)
    norm = mcolors.Normalize(vmin=w_min, vmax=w_max)

    # Subsampling se houver muitos pontos (>500 mil) para não travar o plot
    N = len(uz_all)
    if N > 500_000:
        idx = np.random.choice(N, 500_000, replace=False)
        print(f"  Subsampling: {N:,} → 500,000 pontos para visualização")
        x = uz_all[idx]
        y = fv_all[idx]
        c = w_all[idx]
    else:
        x, y, c = uz_all, fv_all, w_all

    # Ordena por umidade para que pontos azuis (úmidos) fiquem na frente
    order = np.argsort(c)[::-1]
    sc = ax.scatter(
        x[order], y[order],
        c=c[order],
        cmap=cmap,
        norm=LogNorm(vmin=c[order].min(), vmax=c[order].max()),
        s=2,
        alpha=0.25,
        linewidths=0,
        rasterized=True
    )

    # Eixo Y em escala log
    ax.set_yscale('log')

    cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.035)
    cbar.solids.set_alpha(1)
    cbar.update_normal(sc)
    cbar.set_label('Umidade do solo (m³ m⁻³)', fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    # Labels e título
    ax.set_xlabel('Velocidade do vento (m s⁻¹)', fontsize=11)
    y_label = ('Taxa de emissão (g s⁻¹ m⁻²)'
               if area_normalized else 'Fluxo vertical PM10\n(g s⁻¹)')
    ax.set_ylabel(y_label, fontsize=11)

    # Grid suave
    ax.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)

    # Estatísticas no canto
    stats_txt = (
        f'N = {N:,} pts\n'
        f'Vento: {uz_all.min():.1f}–{uz_all.max():.1f} m/s\n'
        f'Umidade: {w_all.min():.3f}–{w_all.max():.3f} m³/m³'
    )
    ax.text(0.97, 0.03, stats_txt,
            transform=ax.transAxes,
            fontsize=7, va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    fig.tight_layout()
    os.makedirs(os.path.dirname(outfig), exist_ok=True)
    fig.savefig(outfig, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Figura salva em: {outfig}")

def make_scatter(uz_all, w_all, fv_all, outfig, area_normalized=True):
    """
    Figura 1 — Scatter principal (um ponto por timestep×pixel).
      X = vento u10 (m/s)
      Y = Fvtot normalizado (g s⁻¹ m⁻²) — escala log
      Cor = quartil de umidade do solo (discreto):
            P0–25  → vermelho   (#d62728)
            P25–50 → amarelo    (#f5c518)
            P50–75 → verde      (#2ca02c)
            P75–100 → azul      (#1f77b4)
    """
    # ---- limites de percentil -------------------------------------------------
    p25, p50, p75 = np.nanpercentile(w_all, [25, 50, 75])
    quartile_colors  = ['#d62728', '#f5c518', '#2ca02c', '#1f77b4']
    quartile_labels  = [
        f'P0–25  (SMOIS ≤ {p25:.3f})',
        f'P25–50 ({p25:.3f} < SMOIS ≤ {p50:.3f})',
        f'P50–75 ({p50:.3f} < SMOIS ≤ {p75:.3f})',
        f'P75–100 (SMOIS > {p75:.3f})',
    ]
    # Atribui índice de quartil a cada ponto (0-3)
    quartile_idx = np.zeros(len(w_all), dtype=int)
    quartile_idx[w_all >  p25] = 1
    quartile_idx[w_all >  p50] = 2
    quartile_idx[w_all >  p75] = 3

    # ---- subsampling ----------------------------------------------------------
    N = len(uz_all)
    if N > 500_000:
        idx = np.random.choice(N, 500_000, replace=False)
        print(f"  Subsampling scatter: {N:,} → 500,000 pontos")
        x = uz_all[idx];  y = fv_all[idx];  qi = quartile_idx[idx]
    else:
        x, y, qi = uz_all, fv_all, quartile_idx

    # Ordena para que quartis mais úmidos (azul) fiquem na frente
    order = np.argsort(qi)
    x, y, qi = x[order], y[order], qi[order]

    # ---- figura ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)

    for q in range(4):
        mask = qi == q
        ax.scatter(
            x[mask], y[mask],
            color=quartile_colors[q],
            s=2, alpha=0.25, linewidths=0, rasterized=True,
            zorder=q + 1
        )

    ax.set_yscale('log')
    ax.set_xlabel('Velocidade do vento (m s⁻¹)', fontsize=11)
    y_label = 'Taxa de emissão (g s⁻¹ m⁻²)' if area_normalized else 'Fluxo vertical PM10 (g s⁻¹)'
    ax.set_ylabel(y_label, fontsize=11)
    ax.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)

    # Legenda manual com patches coloridos
    legend_patches = [
        mpatches.Patch(color=quartile_colors[q], label=quartile_labels[q])
        for q in range(4)
    ]
    ax.legend(handles=legend_patches, title='Quartil de umidade',
              fontsize=7, title_fontsize=8, loc='upper left',
              framealpha=0.8, edgecolor='grey')

    stats_txt = (
        f'N = {N:,} pts\n'
        f'Vento: {uz_all.min():.1f}–{uz_all.max():.1f} m/s\n'
        f'Umidade: {w_all.min():.3f}–{w_all.max():.3f} m³/m³'
    )
    ax.text(0.97, 0.03, stats_txt,
            transform=ax.transAxes, fontsize=7, va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    fig.tight_layout()
    os.makedirs(os.path.dirname(outfig), exist_ok=True)
    fig.savefig(outfig, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Figura 1 salva em: {outfig}")
'''
def make_kde_soilmoisture_threshold(
    uz_all, w_all, fv_all, outfig,
    threshold=0.4,
    area_normalized=True
):
    """
    KDE 2D em um único gráfico.

    Vermelho: SMOIS < threshold
    Azul:     SMOIS >= threshold

    X = vento u10
    Y = emissão (escala log)
    """

    from scipy.stats import gaussian_kde
    from matplotlib.colors import LinearSegmentedColormap

    y_label = (
        'Taxa de emissão (g s⁻¹ m⁻²)'
        if area_normalized
        else 'Fluxo vertical PM10 (g s⁻¹)'
    )

    fig, ax = plt.subplots(
        figsize=(10, 8),
        dpi=150
    )

    grupos = [
        {
            'mask': w_all < threshold,
            'color': '#d62728',
            'label': f'SMOIS < {threshold:.2f} m³/m³'
        },
        {
            'mask': w_all >= threshold,
            'color': '#1f77b4',
            'label': f'SMOIS ≥ {threshold:.2f} m³/m³'
        }
    ]

    # ------------------------------------------------------
    # limites globais para ambas KDEs usarem a mesma grade
    # ------------------------------------------------------
    mask_global = (
        (fv_all > 0)
        & np.isfinite(uz_all)
        & np.isfinite(fv_all)
        & np.isfinite(w_all)
    )

    x_all = uz_all[mask_global]
    y_all = fv_all[mask_global]

    log_y_all = np.log10(y_all)

    x_min = np.nanpercentile(x_all, 0.5)
    x_max = np.nanpercentile(x_all, 99.5)

    ly_min = np.nanpercentile(log_y_all, 0.5)
    ly_max = np.nanpercentile(log_y_all, 99.5)

    xi = np.linspace(x_min, x_max, 150)
    yi = np.linspace(ly_min, ly_max, 150)

    Xi, Yi = np.meshgrid(xi, yi)

    # ------------------------------------------------------
    # KDE para cada grupo
    # ------------------------------------------------------
    for grupo in grupos:

        mask = (
            grupo['mask']
            & (fv_all > 0)
            & np.isfinite(uz_all)
            & np.isfinite(fv_all)
        )

        x_q = uz_all[mask]
        y_q = fv_all[mask]

        n_pts = len(x_q)

        if n_pts < 10:
            continue

        # KDE muito grande fica lenta
        if n_pts > 800000:
            idx = np.random.choice(
                n_pts,
                800000,
                replace=False
            )

            x_q = x_q[idx]
            y_q = y_q[idx]

            print(
                f"  Subsampling: {n_pts:,} → 80,000"
            )

        log_y = np.log10(y_q)

        kde = gaussian_kde(
            np.vstack([x_q, log_y]),
            bw_method='silverman'
        )

        Zi = kde(
            np.vstack([
                Xi.ravel(),
                Yi.ravel()
            ])
        ).reshape(Xi.shape)

        # normalização individual
        Zi = (
            Zi - Zi.min()
        ) / (
            Zi.max() - Zi.min() + 1e-30
        )

        # 1. Colormap transparente → cor (em vez de branco → cor)
        cmap = LinearSegmentedColormap.from_list(
            '',
            [(1, 1, 1, 0), grupo['color']],  # alpha=0 no mínimo → sem sobreposição
            N=256
        )

        # 2. Plotar apenas UMA vez com vmin para cortar ruído de fundo
        cf = ax.contourf(
            Xi,
            10**Yi,
            Zi,
            levels=10,
            cmap=cmap,
            vmin=0.15,   # ignora densidade < 5% → não pinta branco por cima do outro grupo
            alpha=1
        )

        # contornos
        #ax.contour(
        #    Xi,
        #    10**Yi,
        #    Zi,
        #    levels=np.linspace(0.1, 1, 10),
        #    colors=[grupo['color']],
        #    linewidths=1.0
        #)
        
        ax.set_yscale('log')
        ax.set_xlabel('Vento u10 (m s⁻¹)', fontsize=9)
        ax.set_ylabel(y_label, fontsize=9)
        ax.grid(True, which='both', linestyle='--', linewidth=0.3, alpha=0.4)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8)

        # Colorbar lateral por subplot
        cb = fig.colorbar(cf, ax=ax, pad=0.02, fraction=0.04)
        cb.set_label('Densidade KDE (norm.)', fontsize=7)
        cb.ax.tick_params(labelsize=6)

        # Estatísticas
        stats_txt = f'N = {len(x_q):,}'
        ax.text(0.97, 0.97, stats_txt,
                transform=ax.transAxes, fontsize=7, va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    # ------------------------------------------------------
    # layout
    # ------------------------------------------------------
    ax.set_yscale('log')

    ax.set_xlabel(
        'Vento u10 (m s⁻¹)',
        fontsize=11
    )

    ax.set_ylabel(
        y_label,
        fontsize=11
    )

    ax.grid(
        True,
        which='both',
        linestyle='--',
        linewidth=0.4,
        alpha=0.5
    )

    ax.set_axisbelow(True)

    ax.legend(
        [
            plt.Line2D([], [], color='#d62728'),
            plt.Line2D([], [], color='#1f77b4')
        ],
        [
            f'Umidade do solo < {threshold:.2f} m³/m³',
            f'Umidade do solo ≥ {threshold:.2f} m³/m³'
        ],
        loc='upper left',
        fontsize=10
    )

    fig.tight_layout()

    base, ext = os.path.splitext(outfig)
    kde_outfig = f"{base}_kde_threshold{ext}"

    fig.savefig(
        kde_outfig,
        dpi=150,
        bbox_inches='tight'
    )

    plt.close(fig)

    print(f'Figura salva em: {kde_outfig}')
'''

def make_kde_soilmoisture_threshold(
    uz_all, w_all, fv_all, outfig,
    threshold=0.4,
    area_normalized=True
):
    from scipy.stats import gaussian_kde
    from matplotlib.colors import LinearSegmentedColormap
 
    y_label = (
        '$MP_{{10}}$ vertical flux (g s⁻¹ m⁻²)'
        if area_normalized
        else '$MP_{{10}}$ vertical flux (g s⁻¹)'
    )
 
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
 
    grupos = [
        {
            'mask':  w_all < threshold,
            'color': '#d62728',
            'label': f'Soil moisture < {threshold:.2f} m³/m³',
        },
        {
            'mask':  w_all >= threshold,
            'color': '#1f77b4',
            'label': f'Soil moisture ≥ {threshold:.2f} m³/m³',
        }
    ]
 
    # ── Grade global compartilhada ─────────────────────────────────────
    mask_global = (
        (fv_all > 0)
        & np.isfinite(uz_all)
        & np.isfinite(fv_all)
        & np.isfinite(w_all)
    )
    x_all     = uz_all[mask_global]
    log_y_all = np.log10(fv_all[mask_global])
 
    x_min  = np.nanpercentile(x_all,     0.5)
    x_max  = np.nanpercentile(x_all,    99.5)
    ly_min = np.nanpercentile(log_y_all,  0.5)
    ly_max = np.nanpercentile(log_y_all, 99.5)
 
    xi = np.linspace(x_min, x_max,  150)
    yi = np.linspace(ly_min, ly_max, 150)
    Xi, Yi = np.meshgrid(xi, yi)
 
    # ── Acumuladores para usar fora do loop ───────────────────────────
    cf_items  = []   # {'cf': ..., 'color': ..., 'label': ...}
    n_counts  = []
 
    # Níveis e vmin/vmax comuns para as duas nuvens terem intensidade comparável
    levels = np.linspace(0.1, 1, 19)
    vmin, vmax = 0.0, 1.0
 
    # ── Loop principal ────────────────────────────────────────────────
    for grupo in grupos:
        mask = (
            grupo['mask']
            & (fv_all > 0)
            & np.isfinite(uz_all)
            & np.isfinite(fv_all)
        )
        x_q   = uz_all[mask]
        y_q   = fv_all[mask]
        n_pts = len(x_q)
 
        if n_pts < 10:
            continue
 
        if n_pts > 80_000:
            idx = np.random.choice(n_pts, 80_000, replace=False)
            x_q, y_q = x_q[idx], y_q[idx]
            print(f"  Subsampling: {n_pts:,} → 80,000")
 
        log_y = np.log10(y_q)
 
        kde = gaussian_kde(np.vstack([x_q, log_y]), bw_method='silverman')
        Zi  = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
        Zi  = (Zi - Zi.min()) / (Zi.max() - Zi.min() + 1e-30)
        Zi  = np.where(Zi>0.1,Zi,np.nan)
 
        # Transparente → cor  (sem véu branco sobre o outro grupo)
        cmap = LinearSegmentedColormap.from_list(
            '', [(1, 1, 1, 0), grupo['color']], N=256
        )
 
        cf = ax.contourf(
            Xi, 10**Yi, Zi,
            levels=levels,
            cmap=cmap,
            vmin=0.1, vmax=vmax,   # ← mesma normalização para os dois grupos
            alpha=1
        )
 
        ax.contour(
            Xi, 10**Yi, Zi,
            levels=[0.1, 0.2, 0.4, 0.6, 0.8],
            colors=[grupo['color']],
            linewidths=0.8,
            alpha=0.75
        )
 
        cf_items.append({
            'cf':    cf,
            'color': grupo['color'],
            'label': grupo['label']
        })
        n_counts.append(n_pts)
 
    # ── Layout ────────────────────────────────────────────────────────
    ax.set_yscale('log')
    ax.set_xlabel('10 m wind speed (m s⁻¹)', fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9)
 
    ax.legend(
        [plt.Line2D([], [], color=g['color'], linewidth=2) for g in cf_items],
        [g['label'] for g in cf_items],
        loc='upper left',
        fontsize=10
    )
 
    # ── Caixa única com N total e % seco / % úmido ─────────────────────
    n_total = sum(n_counts)
    if n_total > 0:
        # grupos[0] é sempre "seco" (< threshold), grupos[1] é "úmido" (>= threshold)
        n_seco  = n_counts[0] if len(n_counts) > 0 else 0
        n_umido = n_counts[1] if len(n_counts) > 1 else 0
        pct_seco  = 100 * n_seco  / n_total
        pct_umido = 100 * n_umido / n_total
 
        info_text = (
            f"N = {n_total:,}\n"
            f"$N_{{wet}}$ = {pct_umido:.0f}%\n"
            f"$N_{{dry}}$ = {pct_seco:.0f}%"
        )
        ax.text(
            0.93, 0.97,
            info_text,
            transform=ax.transAxes,
            fontsize=11, va='top', ha='right',
            color='black',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='black', alpha=0.85)
        )
 
    # ── Colorbar única combinada (vermelho + azul lado a lado) ────────
    fig.subplots_adjust(right=0.86)
 
    if len(cf_items) >= 2:
        # Duas colorbars coladas, mesma altura/posição, mesma escala
        cb_width = 0.025
        gap = 0.0
        left0 = 0.84
 
        # Ordem: azul (úmido) à esquerda, vermelho (seco) à direita —
        # ajuste conforme preferir, aqui seguindo a ordem dos grupos
        positions = [
            [left0,                 0.11, cb_width, 0.77],  # grupo 0 (vermelho/seco)
            [left0 + cb_width,      0.11, cb_width, 0.77],  # grupo 1 (azul/úmido)
        ]
 
        ticks_valores = np.arange(0, 1.1, 0.1)
 
        for i, (item, pos) in enumerate(zip(cf_items, positions)):
            cax = fig.add_axes(pos)
            cb = fig.colorbar(item['cf'], cax=cax, ticks=ticks_valores)
            cb.ax.tick_params(labelsize=8)
            # Só o último (mais à direita) mostra ticks/labels numéricos completos
            if i < len(cf_items) - 1:
                cb.ax.set_yticklabels([])
                cb.outline.set_visible(True)
 
        # Label central para o par de colorbars (à direita, fora dos ticks)
        fig.text(
            left0 + cb_width * len(cf_items) + 0.055, 0.5,
            'KDE density (normalized)',
            rotation=90, va='center', ha='center', fontsize=10
        )
    elif len(cf_items) == 1:
        cax = fig.add_axes([0.84, 0.11, 0.03, 0.77])
        cb = fig.colorbar(cf_items[0]['cf'], cax=cax)
        cb.set_label('KDE density (normalized)', fontsize=10)
        cb.ax.tick_params(labelsize=8)
 
    base, ext = os.path.splitext(outfig)
    kde_outfig = f"{base}_kde_threshold{ext}"
    fig.savefig(kde_outfig, dpi=150, bbox_inches='tight')
    plt.close(fig)

def make_kde_subplots(uz_all, w_all, fv_all, outfig, area_normalized=True):
    """
    Figura 2 — 2×2 subplots com densidade de kernel (KDE) gaussiana.
    Cada subplot corresponde a um quartil de umidade:
      (0,0) P0–25   vermelho   (#d62728)
      (0,1) P25–50  amarelo    (#f5c518)
      (1,0) P50–75  verde      (#2ca02c)
      (1,1) P75–100 azul       (#1f77b4)
    X = vento u10, Y = Fvtot normalizado (escala log).
    A densidade KDE é estimada em grade regular (x linear, y log) e
    plotada como contourf.
    """
    p25, p50, p75 = np.nanpercentile(w_all, [25, 50, 90])
    p75 = 0.4
    print(100*(w_all<0.4).mean())
    quartile_bounds = [(0,   p25),
                       (p25, p50),
                       (p50, p75),
                       (p75, np.inf)]
    quartile_colors  = ['#d62728', '#f5c518', '#2ca02c', '#1f77b4']
    quartile_titles  = [
        f'P0–25  (SMOIS ≤ {p25:.3f} m³/m³)',
        f'P25–50 ({p25:.3f}–{p50:.3f} m³/m³)',
        f'P50–75 ({p50:.3f}–{p75:.3f} m³/m³)',
        f'P75–100 (SMOIS > {p75:.3f} m³/m³)',
    ]
    positions = [(0, 0),(0, 1),(1, 0),(1, 1)]

    y_label = '$MP_{{10}}$ vertical flux (g s⁻¹ m⁻²)' if area_normalized else '$MP_{{10}}$ vertical flux (g s⁻¹)'

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=150,
                             sharex=False, sharey=False)

    for q, ((row, col), (wlo, whi)) in enumerate(zip(positions, quartile_bounds)):
        ax = axes[0][0]
        color = quartile_colors[q]

        # Filtra pontos do quartil
        mask = (w_all >= wlo) & (w_all < whi) & (fv_all > 0) & np.isfinite(uz_all) & np.isfinite(fv_all)
        x_q = uz_all[mask]
        y_q = fv_all[mask]

        n_pts = len(x_q)
        print(f"  KDE quartil {q+1}: {n_pts:,} pontos")

        if n_pts < 10:
            ax.text(0.5, 0.5, 'Dados insuficientes',
                    ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_title(quartile_titles[q], fontsize=9, fontweight='bold', color=color)
            continue

        # Subsampling para KDE (máx 80k — KDE é O(n²))
        if n_pts > 80_000:
            idx_q = np.random.choice(n_pts, 80_000, replace=False)
            print(f"    Subsampling KDE: {n_pts:,} → 80,000")
            x_q = x_q[idx_q]
            y_q = y_q[idx_q]

        # Trabalha em log(y) para simetria da KDE
        log_y_q = np.log10(y_q)

        # Grade de avaliação
        x_min, x_max = np.nanpercentile(x_q, 0.5), np.nanpercentile(x_q, 99.5)
        ly_min, ly_max = np.nanpercentile(log_y_q, 0.5), np.nanpercentile(log_y_q, 99.5)
        xi = np.linspace(x_min, x_max, 120)
        yi = np.linspace(ly_min, ly_max, 120)
        Xi, Yi = np.meshgrid(xi, yi)

        # Estimativa KDE
        kde = gaussian_kde(np.vstack([x_q, log_y_q]),
                           bw_method='silverman')
        Zi = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)

        # Normaliza densidade entre 0 e 1 para comparabilidade visual
        Zi = (Zi - Zi.min()) / (Zi.max() - Zi.min() + 1e-30)

        Zi = Zi ** grupo['gamma']

        # Contourf usando a cor do quartil com transparência crescente
        from matplotlib.colors import LinearSegmentedColormap
        cmap_q = LinearSegmentedColormap.from_list(
            f'q{q}', ['#ffffff', color], N=256
        )
        cf = ax.contourf(Xi, 10**Yi, Zi,
                         levels=20, cmap=cmap_q, alpha=0.9)
        ax.contour(Xi, 10**Yi, Zi,
                   levels=8, colors=[color], linewidths=0.5, alpha=0.6)

        ax.set_yscale('log')
        ax.set_xlabel('Vento u10 (m s⁻¹)', fontsize=9)
        if col == 0:
            ax.set_ylabel(y_label, fontsize=9)
        ax.set_title(quartile_titles[q], fontsize=9, fontweight='bold', color=color)
        ax.grid(True, which='both', linestyle='--', linewidth=0.3, alpha=0.4)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8)

        # Colorbar lateral por subplot
        cb = fig.colorbar(cf, ax=ax, pad=0.02, fraction=0.04)
        cb.set_label('Densidade KDE (norm.)', fontsize=7)
        cb.ax.tick_params(labelsize=6)

        # Estatísticas
        stats_txt = f'N = {len(x_q):,}'
        ax.text(0.97, 0.97, stats_txt,
                transform=ax.transAxes, fontsize=7, va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    fig.suptitle(
        'Densidade de kernel por quartil de umidade do solo\n'
        'Ressuspensão eólica BR_12km — 2023',
        fontsize=12, fontweight='bold', y=1.01
    )
    fig.tight_layout()

    # Deriva o caminho da figura KDE a partir do scatter principal
    base, ext = os.path.splitext(outfig)
    kde_outfig = f"{base}_kde{ext}"
    fig.savefig(kde_outfig, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figura 2 (KDE) salva em: {kde_outfig}")


# =============================================================================
# Entrypoint
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Scatter plot de sensibilidade do modelo windBlowDust — 2023'
    )
    parser.add_argument('--windBlowDustFolder',
                        default='/home/lcqar/BRAIN/emis/windBlowDustBR',
                        help='Pasta raiz do módulo windBlowDust')
    parser.add_argument('--wrfoutFolder',
                        required=True,
                        help='Pasta com os arquivos wrfout')
    parser.add_argument('--domain',
                        default='2',
                        help='Número do domínio WRF (ex: 2 → d02)')
    parser.add_argument('--GDNAM',
                        default='BR_12km',
                        help='Nome da grade MCIP')
    parser.add_argument('--year',
                        type=int,
                        default=2023,
                        help='Ano de referência')
    parser.add_argument('--outfig',
                        default=None,
                        help='Caminho completo do arquivo de figura de saída (.png)')
    args = parser.parse_args()

    outfolder = os.path.join(args.windBlowDustFolder, 'Outputs', args.GDNAM)

    if args.outfig is None:
        args.outfig = os.path.join(outfolder, 'figures',
                                   f'scatter_sensitivity_{args.year}.png')

    print('=' * 60)
    print(f'  windBlowDustFolder : {args.windBlowDustFolder}')
    print(f'  wrfoutFolder       : {args.wrfoutFolder}')
    print(f'  domain             : {args.domain}')
    print(f'  GDNAM              : {args.GDNAM}')
    print(f'  Outputs folder     : {outfolder}')
    print(f'  Figura de saída    : {args.outfig}')
    print('=' * 60)

    # Descobre arquivos
    pm10_files  = find_pm10_files(outfolder, year=args.year)
    wrf_files   = find_wrf_files(args.wrfoutFolder, args.domain, year=args.year)

    if not pm10_files:
        raise FileNotFoundError(
            f"Nenhum arquivo PM10 encontrado em {outfolder} para {args.year}.\n"
            f"Padrão esperado: windBlowDust_PM10_{args.year}-MM-DD-*"
        )
    if not wrf_files:
        raise FileNotFoundError(
            f"Nenhum arquivo WRF encontrado em {args.wrfoutFolder} para {args.year}.\n"
            f"Padrão esperado: wrfout_d0{args.domain}_{args.year}-MM-DD_*"
        )

    # Coleta dados anuais
    print('\n--- Coletando dados ---')
    uz_all, w_all, fv_all = collect_annual_data(
        pm10_files, wrf_files, outfolder, args.GDNAM
    )

    # Gera o scatter
    print('\n--- Gerando figuras ---')
    area_total_check = load_area_total(outfolder, args.GDNAM)
    area_norm = (area_total_check is not None)
    '''
    make_scatter(uz_all, w_all, fv_all,
                 outfig=args.outfig,
                 area_normalized=area_norm)
    make_kde_subplots(uz_all, w_all, fv_all,
                      outfig=args.outfig,
                      area_normalized=area_norm)
    make_scatter_2(uz_all, w_all, fv_all, os.path.join(outfolder, 'figures', f'scatter_sensitivity_2023_log.png'), area_normalized=True)
    '''
    make_kde_soilmoisture_threshold( uz_all, w_all, fv_all, args.outfig, area_normalized=area_norm)

    print('\nConcluído.')


if __name__ == '__main__':
    main()
