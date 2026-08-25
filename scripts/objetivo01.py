"""
=====================================================================
sensitivity_uncertainty_wbd.py
=====================================================================
 
Análise de Sensibilidade (Índices de Sobol via método de Saltelli) e
Análise de Incerteza (Monte Carlo) para o modelo de ressuspensão
eólica de material particulado (Wind Blown Dust – WBD).
 
Variáveis de entrada avaliadas:
    clayRegrid, siltRegrid, sandRegrid, sRef, ustar, ustarThreshold,
    z0, w, uz, w' (w_prime), alpha, fm, fr
 
Outputs avaliados:
    Fh  – fluxo horizontal total (Fhtot)  [µg/(m·s)]
    Fv  – fluxo vertical  total (Fvtot)   [µg/(m²·s)]
 
Dependências: apenas numpy, matplotlib, scipy (padrão científico).
              NÃO requer SALib – Saltelli sampler e estimadores de
              Sobol implementados internamente.
 
Uso:
    python sensitivity_uncertainty_wbd.py
 
@author: gerado a partir de windBlowDustCalc.py / metPrep.py
"""
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.integrate import trapezoid, cumulative_trapezoid
 
# =====================================================================
# 1. MODELO WBD PARAMETRIZADO
# =====================================================================
 
def _ustar_calc(uz, z0):
    k, z = 0.4, 10.0
    z0 = np.maximum(z0, 1e-6)
    return k * uz / np.log(z / z0)
 
 
def _ustarTd_calc(D):
    An, gama, roa, rop, g = 0.0123, 5e-4, 1.227, 2665.0, 9.81
    return np.sqrt(An * (rop * g * (D * 1e-6) / roa + gama / (roa * D * 1e-6)))
 
 
def _fm_calc(w, clay):
    wl = (0.0014 * clay**2 + 0.17 * clay) / 100.0
    fm = (1.0 + 1.21 * np.maximum(w - wl, 0.0) ** 0.68) ** 0.5
    return np.where(w < wl, 1.0, fm)
 
 
def _fr_calc(av):
    sigmaV, mV, betaV = 1.45, 0.16, 202.0
    sigmaS, mS, betaS = 1.0,  0.5,  90.0
    av = np.minimum(av, 0.046)
    alphaV = -0.35 * np.log(1.0 - np.maximum(av, 1e-8))
    alphaS = 0.02
    t1 = 1.0 - sigmaV * mV * alphaV
    t2 = 1.0 + betaV  * mV * alphaV
    t3 = 1.0 - sigmaS * mS * (alphaS / np.maximum(1.0 - av, 1e-8))
    t4 = 1.0 + betaS  * mS * (alphaS / np.maximum(1.0 - av, 1e-8))
    return np.sqrt(np.maximum(t1 * t2 * t3 * t4, 0.0))
 
 
def _alpha_calc(clay,silt,sand,pb,ustar):
    
    g = 9.81 # m/s²
    
    Ca = 0.001*sand+0.0006*silt+0.0002*clay # sem unidade
    Cb = 1.37 # sem unidade
    
    p = 5000*sand+10000*silt+30000*clay #N/m²
    
    pp = 2665 #kg/m³
    #pb kg/m³
    #ustar m/s
    
    alpha = ((Ca*g*clay*pb)/(2*p))*(0.24+Cb*ustar*np.sqrt(pp/p))
        
    return alpha

def _sRef_calc(clay,silt,sand,Dp):
    MMD = [210, 125, 2]
    desv = [1.6, 1.8, 2]
    
    sand_2 = sand/(sand+silt+clay)
    silt_2 = silt/(sand+silt+clay)
    clay_2 = clay/(sand+silt+clay)
    
    M = [sand_2,silt_2,clay_2]
    pp = 2655

    dM_dln = []

    for j in range(0,len(MMD)):
        print(j)

        t1 = M[j][:,None]
        t2 = np.exp((np.log(Dp[None,:])-np.log(MMD[j]))**2/(-2*np.log(desv[j])**2))
        t3 = np.sqrt(2*np.pi)*np.log(desv[j])
        
        dM = t1*t2/t3

        dM_dln.append(dM)

    dM_dln = np.nansum(dM_dln,axis=0)
    dS = dM_dln / ((2/3) * pp * Dp[None,:]**2)
    s_total = trapezoid(dS, Dp, axis=1)
    dS_rel = dS / s_total[:,None]
    
    return dS_rel,sand_2,silt_2,clay_2
 
def wbd_model(X, D):
    """
    Avalia o modelo WBD para matriz X (n_amostras x 13).
 
    Colunas de X:
        0  clay           [%]         clayRegrid
        1  silt           [%]         siltRegrid
        2  sand           [%]         sandRegrid
        3  sRef           [-]
        4  ustar          [m/s]
        5  ustarThreshold [m/s]
        6  z0             [m]
        7  w              [m3/m3]
        8  uz             [m/s]
        9  w_prime        [m3/m3]
       10  alpha          [-]
       11  fm             [-]
       12  fr             [-]
 
    Retorna: Fh (Fhtot), Fv (Fvtot)  [ug/(m2·s)]
    """
    clay  = X[:, 0]
    silt  = X[:, 1]
    sand  = X[:, 2]
    z0    = X[:, 3]
    pb    = X[:, 4]
    uz    = X[:, 5]
    fm_in = X[:, 6]
    fr_in = X[:, 7]
 
    g, roa, c = 9.81, 1227.0, 1.0
    
    dS_rel,sand,silt,clay = _sRef_calc(clay,silt,sand,D)
 
    S_acumulada = cumulative_trapezoid(dS_rel, D, axis=1, initial=0) * 100
    S_bins = np.diff(S_acumulada, axis=1)
 
    Fhtot = []

    ustar = _ustar_calc(uz, z0)    

    for j,diameter in enumerate(D[1:]):
        
        sRef = S_bins[:,j]
     
        ustarTD = _ustarTd_calc(diameter)
        
        ustarT = ustarTD * fm_in * fr_in
     
        ratio = np.where(ustar > 0, ustarT / ustar, np.inf)
        Fhd = (c * roa * ustar**3 / g) * (1.0 - ratio) * (1.0 + ratio)**2
        Fhd = np.where((ustarT > ustar) | (Fhd < 0), 0.0, Fhd)
     
        Fhtot.append(Fhd * sRef * 1e6 * 1e-2)
 
    Fhtot = np.stack(Fhtot)
    Fhtot = np.nansum(Fhtot,axis=0)
 
    alpha = _alpha_calc(clay,silt,sand,pb,ustar)
    Fvtot = alpha * Fhtot
 
    return Fhtot, Fvtot, clay, silt, sand
 
 
# =====================================================================
# 2. ESPAÇO DE PARÂMETROS
# =====================================================================
 
PARAM_NAMES = [
    'clayRegrid',
    'siltRegrid',
    'sandRegrid',
    'z0',
    'pb',
    'uz',
    'fm',
    'fr']
 
BOUNDS = np.array([
    [0,     1],
    [0,     1],
    [0,     1],
    [0.000292,  0.00156],
    [1000,  1800],
    [0,     14.0],
    [1.0,   1.487],
    [1.37,  1.71],
])

NUM_VARS = len(PARAM_NAMES)

PARAM_NAMES_FIGS = [
    r'$f_{clay}$',
    r'$f_{silt}$',
    r'$f_{sand}$',
    r'$z_0$',
    r'$\rho_b$',
    r'$u_{10}$',
    r'$f_m$',
    r'$f_r$']

 
# =====================================================================
# 3. SALTELLI SAMPLER (sem SALib)
# =====================================================================
 
def saltelli_sample(bounds, N, seed=42):
    """
    Gera amostras pelo esquema de Saltelli para análise de Sobol.
    Total de linhas = N * (2*k + 2), onde k = num_vars.
    """
    rng = np.random.default_rng(seed)
    k = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
 
    def lhs(n, d):
        M = np.zeros((n, d))
        for j in range(d):
            M[:, j] = (rng.permutation(n) + rng.random(n)) / n
        return M
 
    A = lhs(N, k)
    B = lhs(N, k)
    scale = lambda M: lo + M * (hi - lo)
 
    rows = [scale(A), scale(B)]
    for i in range(k):
        AB = A.copy(); AB[:, i] = B[:, i]; rows.append(scale(AB))
    for i in range(k):
        BA = B.copy(); BA[:, i] = A[:, i]; rows.append(scale(BA))
 
    return np.vstack(rows)
 
 
# =====================================================================
# 4. ESTIMADORES DE SOBOL (Jansen 1999)
# =====================================================================
 
def sobol_indices(Y_all, N):
    """
    Calcula S1 e ST a partir das saídas do Saltelli sampler.
    Usa bootstrap (500 amostras) para IC 95%.
    """
    k = NUM_VARS
    Y_A  = Y_all[:N]
    Y_B  = Y_all[N:2*N]
    Y_AB = [Y_all[(2 + i)*N:(3 + i)*N]         for i in range(k)]
    Y_BA = [Y_all[(2 + k + i)*N:(3 + k + i)*N] for i in range(k)]
 
    Var_Y = np.var(np.concatenate([Y_A, Y_B]), ddof=1)
 
    S1 = np.zeros(k); ST = np.zeros(k)
    S1c = np.zeros(k); STc = np.zeros(k)
 
    rng_b = np.random.default_rng(0)
    n_boot = 500
    idx_b = rng_b.integers(0, N, (n_boot, N))
 
    for i in range(k):
        yab = Y_AB[i]
        S1[i] = max(0.0, (np.var(Y_B, ddof=1) - 0.5*np.mean((Y_B - yab)**2)) / Var_Y)
        ST[i] = max(0.0, 0.5 * np.mean((Y_A - yab)**2) / Var_Y)
 
        s1b = np.zeros(n_boot); stb = np.zeros(n_boot)
        for b, idx in enumerate(idx_b):
            ya_, yb_, yab_ = Y_A[idx], Y_B[idx], yab[idx]
            vb = np.var(np.concatenate([ya_, yb_]), ddof=1)
            if vb > 0:
                s1b[b] = max(0, (np.var(yb_, ddof=1) - 0.5*np.mean((yb_-yab_)**2)) / vb)
                stb[b] = max(0, 0.5 * np.mean((ya_-yab_)**2) / vb)
        S1c[i] = np.std(s1b, ddof=1) * 1.96
        STc[i] = np.std(stb, ddof=1) * 1.96
 
    return {'S1': S1, 'ST': ST, 'S1_conf': S1c, 'ST_conf': STc}
 
 
# =====================================================================
# 5. ANÁLISE DE SOBOL
# =====================================================================
 
def run_sobol(D, N=int(2**16), seed=42):
    print(f"\n{'='*64}")
    print(f"  ANÁLISE DE SENSIBILIDADE (SOBOL)  |  N={N}  |  D={D} µm")
    print(f"{'='*64}")
 
    X_all = saltelli_sample(BOUNDS, N, seed=seed)
    print(f"  Amostras totais: {X_all.shape[0]}")
 
    Fh_all, Fv_all, clay, silt, sand = wbd_model(X_all, D)
 
    X_all[:, 0] = clay
    X_all[:, 1] = silt
    X_all[:, 2] = sand
    
    Si_Fh = sobol_indices(Fh_all, N)
    Si_Fv = sobol_indices(Fv_all, N)
 
    def print_si(Si, label):
        print(f"\n  Índices de Sobol – {label}:")
        print(f"  {'Variável':<20} {'S1':>8} {'±CI':>8} {'ST':>8} {'±CI':>8}")
        for i, n in enumerate(PARAM_NAMES):
            print(f"  {n:<20} {Si['S1'][i]:>8.4f} {Si['S1_conf'][i]:>8.4f} "
                  f"{Si['ST'][i]:>8.4f} {Si['ST_conf'][i]:>8.4f}")
 
    print_si(Si_Fh, "Fh (Fhtot)")
    print_si(Si_Fv, "Fv (Fvtot)")
    return Si_Fh, Si_Fv
 
 
# =====================================================================
# 6. MONTE CARLO
# =====================================================================
 
def run_montecarlo(D, n_samples=100000, seed=42):
    print(f"\n{'='*64}")
    print(f"  MONTE CARLO  |  n={n_samples}  |  D={D} µm")
    print(f"{'='*64}")
 
    rng = np.random.default_rng(seed)
    lo, hi = BOUNDS[:, 0], BOUNDS[:, 1]
    X_mc = rng.uniform(lo, hi, size=(n_samples, NUM_VARS))
 
    Fh_mc, Fv_mc, clay, silt, sand = wbd_model(X_mc, D)
 
    X_mc[:, 0] = clay
    X_mc[:, 1] = silt
    X_mc[:, 2] = sand
 
    for label, vals in [("Fh (Fhtot) [µg/m²·s]", Fh_mc),
                        ("Fv (Fvtot) [µg/m²·s]", Fv_mc)]:
        vp = vals[vals > 0]
        print(f"\n  {label}")
        print(f"    Amostras c/ emissão : {len(vp)}/{n_samples} ({100*len(vp)/n_samples:.1f}%)")
        if len(vp):
            p025, p50, p975 = np.percentile(vp, [2.5, 50, 97.5])
            print(f"    Média     : {np.mean(vp):.4e}")
            print(f"    Median   : {p50:.4e}")
            print(f"    Desv. pad.: {np.std(vp):.4e}")
            print(f"    IC 95%%   : [{p025:.4e} ; {p975:.4e}]")
 
    return Fh_mc, Fv_mc, X_mc
 
 
# =====================================================================
# 7. FIGURAS
# =====================================================================
 
C_S1 = '#2c7bb6'; C_ST = '#d7191c'
C_FH = '#2c7bb6'; C_FV = '#d7191c'
DPI  = 150
 
 
def plot_sobol(Si_Fh, Si_Fv, savefig=True):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharey=True,
                             facecolor='white')
    fig.subplots_adjust(wspace=0.05, left=0.17, right=0.97,
                        top=0.91, bottom=0.10)
 
    def panel(ax, Si, title,label_legenda,leg):
        y  = np.arange(NUM_VARS)
        s1 = np.clip(Si['S1'], 0, 1)
        st = np.clip(Si['ST'], 0, 1)
        ax.barh(y - 0.18, st, height=0.32, color=C_ST, alpha=0.80,
                xerr=Si['ST_conf'],
                error_kw=dict(ecolor='#6b0000', lw=1.1, capsize=3.5),
                label="$S_T$ (total index)", zorder=3)
        ax.barh(y + 0.18, s1, height=0.32, color=C_S1, alpha=0.80,
                xerr=Si['S1_conf'],
                error_kw=dict(ecolor='#08306b', lw=1.1, capsize=3.5),
                label="$S_1$ (first order)", zorder=4)
        ax.set_yticks(y)
        ax.set_yticklabels(PARAM_NAMES_FIGS, fontsize=12)
        ax.set_xlabel("Sobol index for "+label_legenda, fontsize=12)
        ax.set_title(title, fontsize=12, loc='left')
        ax.axvline(0, color='k', lw=0.7)
        ax.set_xlim(0, max(float(np.max(st)) * 1.3, float(np.max(s1)) * 1.3))
        if leg == True:
            ax.legend(fontsize=10, loc='lower right', framealpha=0.85)
        ax.tick_params(axis='x', labelsize=10)
        ax.grid(axis='x', linestyle='--', alpha=0.35, zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.invert_yaxis()
 
    df_sobol = pd.DataFrame({'Variáveis':PARAM_NAMES_FIGS,
                             'S1_FH': np.clip(Si_Fh['S1'], 0, 1),
                             'S1_FH_desv':Si_Fh['S1_conf'],
                             'ST_FH': np.clip(Si_Fh['ST'], 0, 1),
                             'ST_FH_desv':Si_Fh['ST_conf'],
                             'S1_FV': np.clip(Si_Fv['S1'], 0, 1),
                             'S1_FV_desv':Si_Fv['S1_conf'],
                             'ST_FV': np.clip(Si_Fv['ST'], 0, 1),
                             'ST_FV_desv':Si_Fv['ST_conf']})
 
    df_sobol.to_csv('/home/lcqar/BRAIN/emis/windBlowDustBR/Outputs/BR_12km/tables/obj_01_sobol.csv',index=False)
 
    panel(axes[0], Si_Fh, "(a)", "$F_h$", False)
    panel(axes[1], Si_Fv, "(b)", "$F_v$", True)
    axes[1].set_yticks(np.arange(NUM_VARS))
    axes[1].set_yticklabels(PARAM_NAMES_FIGS, fontsize=12)
    
    #fig.suptitle("Análise de Sensibilidade (Sobol) – Ressuspensão Eólica (WBD)",
    #             fontsize=11, fontweight='bold', y=0.98)
    if savefig:
        fig.savefig('sobol_wbd.png', dpi=DPI, bbox_inches='tight')
        print("\n  Figura salva: sobol_wbd.png")
    return fig
 
 
def plot_heatmap(Si_Fh, Si_Fv, savefig=True):
    fig, ax = plt.subplots(figsize=(6, 5), facecolor='white')
    data = np.column_stack([
        np.clip(Si_Fh['ST'], 0, 1),
        np.clip(Si_Fv['ST'], 0, 1),
    ])
    im = ax.imshow(data, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['$F_h$', '$F_v$'], fontsize=11, fontweight='bold')
    ax.set_yticks(np.arange(NUM_VARS))
    ax.set_yticklabels(PARAM_NAMES, fontsize=9)
    for r in range(NUM_VARS):
        for c in range(2):
            v = data[r, c]
            ax.text(c, r, f"{v:.3f}", ha='center', va='center',
                    fontsize=8, fontweight='bold',
                    color='white' if v > 0.5 else 'black')
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Índice de Sobol Total ($S_T$)", fontsize=9)
    #ax.set_title("Heatmap de Sensibilidade Total – WBD",
    #             fontsize=11, fontweight='bold', pad=8)
    fig.tight_layout()
    if savefig:
        fig.savefig('heatmap_sobol_wbd.png', dpi=DPI, bbox_inches='tight')
        print("  Figura salva: heatmap_sobol_wbd.png")
    return fig
 
 
def plot_montecarlo(Fh_mc, Fv_mc, savefig=True):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor='white')
    fig.subplots_adjust(wspace=0.28, left=0.09, right=0.97,
                        top=0.87, bottom=0.12)
    for ax, vals, label, color in [
        (axes[0], Fh_mc, "$F_h$", C_FH),
        (axes[1], Fv_mc, "$F_v$", C_FV),
    ]:
        vp = vals[vals > 0]
        if label == "$F_h$":
            vp = vp/1e3
        if len(vp) == 0:
            ax.text(0.5, 0.5, 'Sem emissão', transform=ax.transAxes,
                    ha='center', va='center'); ax.set_title(label); continue
        lv = np.log10(vp)
        ax.hist(lv, bins=70, color=color, alpha=0.72,
                edgecolor='white', linewidth=0.2)
        p025, p50, p975 = np.percentile(lv, [2.5, 50, 97.5])
        ax.axvline(p50,  color='black',   lw=1.5, ls='--', label='Median')
        ax.axvline(p025, color='dimgray', lw=1.2, ls=':',  label='IC 2.5%')
        ax.axvline(p975, color='dimgray', lw=1.2, ls=':',  label='IC 97.5%')
        kde_x = np.linspace(lv.min(), lv.max(), 300)
        kde_y = stats.gaussian_kde(lv)(kde_x)
        n_h, bins_h = np.histogram(lv, bins=70)
        bw = np.diff(bins_h).mean()
        #ax.plot(kde_x, kde_y * len(lv) * bw, color='black', lw=1.3, alpha=0.7)
        if label == "$F_h$":
            ax.set_xlabel(f"log₁₀({label})  (mg/m·s)", fontsize=12)
            ax.set_ylabel("Frequency", fontsize=12)
            unidade = "mg/m·s"
            ax.set_title('(a)', fontsize=12, loc='left')
        elif label == "$F_v$":
            unidade = "µg/m²·s"
            ax.legend(fontsize=10)
            ax.set_xlabel(f"log₁₀({label})  (µg/m²·s)", fontsize=12)
            ax.set_title('(b)', fontsize=12, loc='left')
        ax.tick_params(labelsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        md, mu, sd = np.median(vp), np.mean(vp), np.std(vp)
        p2r, p97r = np.percentile(vp, [2.5, 97.5])
        fmt = lambda x: f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        txt = (f"n = {100*len(vp)/len(vals):,.1f}%\n"
           f"μ = {fmt(mu)} {unidade}\n"
           f"x̃ = {fmt(md)} {unidade}\n"
           f"σ = {fmt(sd)} {unidade}\n"
           f"IC₉₅ = [{fmt(p2r)} ; {fmt(p97r)}] {unidade}")
        ax.text(0.03, 0.85, txt, transform=ax.transAxes, ha='left', va='center',
                fontsize=10,
                bbox=dict(facecolor='white', alpha=0.80, edgecolor='gray',
                          boxstyle='round,pad=0.35'))

    #fig.suptitle("Análise de Incerteza (Monte Carlo) – Ressuspensão Eólica (WBD)",
    #             fontsize=11, fontweight='bold', y=0.99)
    if savefig:
        fig.savefig('montecarlo_wbd.png', dpi=DPI, bbox_inches='tight')
        print("  Figura salva: montecarlo_wbd.png")
    return fig
 
 
def plot_scatter(X_mc, Fh_mc, Fv_mc, savefig=True):
    mask = (Fh_mc > 0) & (Fv_mc > 0)
    fig, axes = plt.subplots(2, NUM_VARS,
                             figsize=(2.4 * NUM_VARS, 5.0),
                             facecolor='white')
    fig.subplots_adjust(wspace=0.30, hspace=0.50,
                        left=0.04, right=0.99, top=0.89, bottom=0.12)
    row_info = [
        (np.log10(Fh_mc[mask] + 1e-30), 'log₁₀(Fh)', C_FH),
        (np.log10(Fv_mc[mask] + 1e-30), 'log₁₀(Fv)', C_FV),
    ]
    for col, name in enumerate(PARAM_NAMES):
        for row, (yvals, ylabel, color) in enumerate(row_info):
            ax = axes[row, col]
            ax.scatter(X_mc[mask, col], yvals, s=1.5, alpha=0.12,
                       color=color, rasterized=True)
            try:
                m, b, r, *_ = stats.linregress(X_mc[mask, col], yvals)
                xp = np.linspace(X_mc[mask, col].min(), X_mc[mask, col].max(), 50)
                ax.plot(xp, m*xp + b, color='black', lw=1.0, alpha=0.8)
                ax.set_title(f"r={r:.2f}", fontsize=6, pad=2)
            except Exception:
                pass
            ax.set_xlabel(name, fontsize=6.5)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=7)
            ax.tick_params(labelsize=5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    fig.suptitle("Scatter – Variáveis de Entrada vs. Fh e Fv  (r = Pearson)",
                 fontsize=9.5, fontweight='bold')
    if savefig:
        fig.savefig('scatter_sensitivity_wbd.png', dpi=DPI, bbox_inches='tight')
        print("  Figura salva: scatter_sensitivity_wbd.png")
    return fig
 
 
# =====================================================================
# 8. EXECUÇÃO PRINCIPAL
# =====================================================================
 
if __name__ == "__main__":
 
    # ---- Configurações ----
    
    p1 = np.arange(0.1, 1.0, 0.1)    # 0.1 a 0.9
    p2 = np.arange(1.0, 10.0, 1.0)   # 1 a 9
    p3 = np.arange(10.0, 100.0, 10.0) # 10 a 90
    p4 = np.arange(100.0, 1000.0, 100.0) # 100 a 900
    p5 = np.arange(1000.0, 2000.0 + 1000.0, 1000.0) # 1000 a 2000
    Dp = np.concatenate([p1, p2, p3, p4, p5])
    
    D_PARTICLE   = Dp     # diâmetro de partícula referência [µm]
    N_SOBOL      = int(2**16)   # amostras base Sobol (>= 512 recomendado)
    N_MONTECARLO = 200000  # amostras Monte Carlo
    SEED         = 42
    SHOW_PLOTS   = True   # False em ambiente sem display (servidor)
    SAVE_FIGS    = True
 
    if not SHOW_PLOTS:
        import matplotlib
        matplotlib.use('Agg')
 
    # Sobol
    Si_Fh, Si_Fv = run_sobol(D_PARTICLE, N=N_SOBOL, seed=SEED)
    plot_sobol(Si_Fh, Si_Fv, savefig=SAVE_FIGS)
    #plot_heatmap(Si_Fh, Si_Fv, savefig=SAVE_FIGS)
 
    # Monte Carlo
    Fh_mc, Fv_mc, X_mc = run_montecarlo(D_PARTICLE, n_samples=N_MONTECARLO,
                                         seed=SEED)
    plot_montecarlo(Fh_mc, Fv_mc, savefig=SAVE_FIGS)
    #plot_scatter(X_mc, Fh_mc, Fv_mc, savefig=SAVE_FIGS)
 
    if SHOW_PLOTS:
        plt.show()
 
    print("\n" + "="*64)
    print("  Arquivos gerados:")
    print("    sobol_wbd.png               – S1 e ST por variável (Fh e Fv)")
    print("    heatmap_sobol_wbd.png       – Heatmap ST comparativo")
    print("    montecarlo_wbd.png          – Distribuição de Fh e Fv")
    print("    scatter_sensitivity_wbd.png – Scatter entradas vs. saídas")
    print("="*64)
