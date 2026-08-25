from netCDF4 import Dataset
from matplotlib.colors import LogNorm
import os
import numpy as np
import xarray as xr
import glob
import matplotlib.pyplot as plt

###########################################################################

meses = ['01','02','03','04','05','06','07','08','09','10','11','12']

path = "/home/lcqar/BRAIN/emis/windBlowDustBR/Outputs/BR_12km/"

###########################################################################

dic = {}
'''
for mes in meses:

    path_mes = path + "2021_"+mes+"/windBlowDust_PM10_2021-"+mes+"*.nc"

    arquivos = sorted(glob.glob(path_mes))

    ds = xr.open_mfdataset(
        arquivos,
        engine = "netcdf4",
        concat_dim="TSTEP",
        combine="nested")

    pm10 = ds["PM10"]

    pm10_soma = pm10.max(dim="TSTEP")

    array_2d = pm10_soma.values

    print(np.max(array_2d))

    print(array_2d)
    print(type(array_2d))

    dic[mes] = array_2d
    vmin = 1e-3
    vmax = float(pm10_soma.max())

    pm10_soma.plot(
        norm=LogNorm(vmin=vmin,vmax=vmax),
        cmap="viridis",
        cbar_kwargs={"label":"PM10"})

    plt.title(f"PM10 Acumulado - {mes}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    plt.savefig("/home/lcqar/BRAIN/emis/windBlowDustBR/Outputs/BR_12km/figures/PM10_max_"+mes+".png")

    plt.close()
'''
###########################################################################

arr_pix = []

for mes in meses:

    path_mes = path + "windBlowDust_PM10_2023-"+mes+"*.nc"

    arquivos = sorted(glob.glob(path_mes))

    ds = xr.open_mfdataset(
        arquivos,
        engine = "netcdf4",
        concat_dim="TSTEP",
        combine="nested")

    pm10 = ds["PM10"]

    pm10_soma = pm10.sum(dim="TSTEP")

    array_2d = pm10_soma.values

    print(array_2d)
    print(type(array_2d))

    arr_pix.append(np.sum(array_2d>0))
    print(arr_pix)
    dic[mes] = array_2d
    vmin = 1e-3
    vmax = float(pm10_soma.max())

    plt.figure(figsize=(10,8))

    pm10_soma.plot(
        norm=LogNorm(vmin=vmin,vmax=vmax),
        cmap="viridis",
        cbar_kwargs={"label":"PM10"})

    plt.title(f"PM10 Acumulado - {mes}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    plt.savefig("/home/lcqar/BRAIN/emis/windBlowDustBR/Outputs/BR_12km/objetivo02/PM10_"+mes+".png")

    plt.close()

###########################################################################

import numpy as np

# Organiza os meses
meses = sorted(dic.keys())

# Empilha os arrays
stack = np.stack([dic[m] for m in meses], axis=0)

# Índice do maior valor
idx_max = np.argmax(stack, axis=0)
idx_min = np.argmin(stack, axis=0)

# Converte para mês (1 a 12)
resultado_max = idx_max + 1
resultado_min = idx_min + 1

# Máscara onde todos são zero
mask_zero = np.all(stack == 0, axis=0)

# Coloca 13 onde tudo é zero
resultado_max[mask_zero] = 13
resultado_min[mask_zero] = 13

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm

# Lista de cores (1 a 13)
cores = [
    "darkred",      # 1
    "red",          # 2
    "orange",       # 3

    "lightblue",    # 4
    "paleturquoise",# 5
    "lightcyan",    # 6

    "dodgerblue",   # 7
    "blue",         # 8
    "navy",         # 9

    "lightsalmon",  # 10
    "peachpuff",    # 11
    "lightpink",    # 12

    "lightgray"     # 13
]

# Cria colormap categórico
cmap = ListedColormap(cores)

# Define limites das categorias
bounds = np.arange(0.5, 14.5, 1)
norm = BoundaryNorm(bounds, cmap.N)

plt.figure(figsize=(10, 8))

img = plt.imshow(
    resultado_max,
    cmap=cmap,
    norm=norm,
    origin="lower"
)

# Barra de cores com ticks discretos
cbar = plt.colorbar(img, ticks=range(1,14))
cbar.set_label("Mês do maior valor")

plt.title("Mês com maior valor por ponto")
plt.xlabel("Coluna")
plt.ylabel("Linha")

plt.savefig(
    "/home/lcqar/BRAIN/emis/windBlowDustBR/Outputs/BR_12km/objetivo02/PM10_max_meses.png",
    dpi=300,
    bbox_inches="tight"
)

plt.figure(figsize=(10, 8))

img = plt.imshow(
    resultado_min,
    cmap=cmap,
    norm=norm,
    origin="lower"
)

# Barra de cores com ticks discretos
cbar = plt.colorbar(img, ticks=range(1,14))
cbar.set_label("Mês do menor valor")

plt.title("Mês com menor valor por ponto")
plt.xlabel("Coluna")
plt.ylabel("Linha")

plt.savefig(
    "/home/lcqar/BRAIN/emis/windBlowDustBR/Outputs/BR_12km/objetivo02/PM10_min_meses.png",
    dpi=300,
    bbox_inches="tight"
)


# Expande dimensões para indexação
idx_max_exp = idx_max[None, :, :]
idx_min_exp = idx_min[None, :, :]

# Pega os valores reais nos meses de max e min
val_max = np.take_along_axis(stack, idx_max_exp, axis=0)[0]
val_min = np.take_along_axis(stack, idx_min_exp, axis=0)[0]

# Amplitude real (diferença dos valores)
resultado_amplitude = val_max - val_min
resultado_razao_max = resultado_amplitude/val_max
resultado_razao_min = resultado_amplitude/val_min
resultado_razao_med = val_max/np.mean(stack,axis=0)
resultado_razao = val_max/val_min

# Onde tudo é zero, força 0 (ou NaN, se preferir)
resultado_amplitude[mask_zero] = 0
resultado_razao_max[mask_zero] = 0
resultado_razao_min[mask_zero] = 0
resultado_razao_med[mask_zero] = 0
resultado_razao[mask_zero] = 0

resultado_amplitude[resultado_amplitude <= 0] = np.nan
resultado_razao_max[resultado_razao_max <= 0] = np.nan
resultado_razao_min[resultado_razao_min <= 0] = np.nan
resultado_razao_med[resultado_razao_med <= 0] = np.nan
resultado_razao[resultado_razao <= 0] = np.nan

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

plt.figure(figsize=(10, 8))

img = plt.imshow(
    resultado_amplitude,
    origin="lower",
    cmap="jet",
    norm=LogNorm()   # <<< escala log
)

cbar = plt.colorbar(img)
cbar.set_label("Amplitude (escala log)")

plt.title("Amplitude sazonal (Max - Min) em escala log")
plt.xlabel("Coluna")
plt.ylabel("Linha")

plt.savefig(
    "/home/lcqar/BRAIN/emis/windBlowDustBR/Outputs/BR_12km/objetivo02/PM10_amplitude_jet.png",
    dpi=300,
    bbox_inches="tight"
)

def plot_log_jet(data, titulo, arquivo, label):
    
    # Cópia segura
    plot_data = data.copy()

    # Remove inválidos
    plot_data[plot_data <= 0] = np.nan

    plt.figure(figsize=(10, 8))

    img = plt.imshow(
        plot_data,
        origin="lower",
        cmap="jet",
        norm=LogNorm()
    )

    cbar = plt.colorbar(img)
    cbar.set_label(label)

    plt.title(titulo)
    plt.xlabel("Coluna")
    plt.ylabel("Linha")

    plt.savefig(
        arquivo,
        dpi=300,
        bbox_inches="tight"
    )

# Razão (max/min)
plot_log_jet(
    resultado_razao,
    "Razão Max / Min (escala log)",
    "/home/lcqar/BRAIN/emis/windBlowDustBR/Outputs/BR_12km/objetivo02/PM10_razao_max_min.png",
    "Max / Min (log)"
)

# (max - min) / max
plot_log_jet(
    resultado_razao_max,
    "Razão (Max - Min) / Max (escala log)",
    "/home/lcqar/BRAIN/emis/windBlowDustBR/Outputs/BR_12km/objetivo02/PM10_razao_amp_max.png",
    "(Max - Min) / Max (log)"
)

# (max - min) / min
plot_log_jet(
    resultado_razao_min,
    "Razão (Max - Min) / Min (escala log)",
    "/home/lcqar/BRAIN/emis/windBlowDustBR/Outputs/BR_12km/objetivo02/PM10_razao_amp_min.png",
    "(Max - Min) / Min (log)"
)

# max / med
plot_log_jet(
    resultado_razao_med,
    "Razão Max / Med (escala log)",
    "/home/lcqar/BRAIN/emis/windBlowDustBR/Outputs/BR_12km/objetivo02/PM10_razao_amp_med.png",
    "Max / Med (log)"
)

# Remove o 13
arr_filtrado = resultado_max[resultado_max != 13]

# Valores de 1 a 12
x = np.arange(1, 13)

# Conta frequência
contagem = [(arr_filtrado == i).sum() for i in x]

# Cria figura e eixo
fig, ax = plt.subplots(figsize=(10, 5))

# Plot
ax.bar(x, contagem, color=cores[:12])

ax.set_xticks(x)
ax.set_xlabel("Valores")
ax.set_ylabel("Frequência")
ax.set_title("Frequência dos valores de 1 a 12")

ax.grid(axis="y", linestyle="--", alpha=0.5)

# Salva a figura
fig.savefig(
    "/home/lcqar/BRAIN/emis/windBlowDustBR/Outputs/BR_12km/objetivo02/frequencia_max_1_a_12.png",
    dpi=300,
    bbox_inches="tight"
)

# Remove o 13
arr_filtrado = resultado_min[resultado_min != 13]

# Valores de 1 a 12
x = np.arange(1, 13)

# Conta frequência
contagem = [(arr_filtrado == i).sum() for i in x]

# Cria figura e eixo
fig, ax = plt.subplots(figsize=(10, 5))

# Plot
ax.bar(x, contagem, color=cores[:12])

ax.set_xticks(x)
ax.set_xlabel("Valores")
ax.set_ylabel("Frequência")
ax.set_title("Frequência dos valores mínimos de 1 a 12")

ax.grid(axis="y", linestyle="--", alpha=0.5)

# Salva a figura
fig.savefig(
    "/home/lcqar/BRAIN/emis/windBlowDustBR/Outputs/BR_12km/objetivo02/frequencia_min_1_a_12.png",
    dpi=300,
    bbox_inches="tight"
)


# Valores de 1 a 12
x = np.arange(1, 13)

# Conta frequência
contagem = arr_pix

# Cria figura e eixo
fig, ax = plt.subplots(figsize=(10, 5))

# Plot
ax.bar(x, contagem, color=cores[:12])

ax.set_xticks(x)
ax.set_xlabel("Valores")
ax.set_ylabel("Frequência")
ax.set_title("Frequência de pixels com valor acima de 0 de 1 a 12")

ax.grid(axis="y", linestyle="--", alpha=0.5)

# Salva a figura
fig.savefig(
    "/home/lcqar/BRAIN/emis/windBlowDustBR/Outputs/BR_12km/objetivo02/frequencia_maior_0_1_a_12.png",
    dpi=300,
    bbox_inches="tight"
)


'''

lista_meses = []

for mes in meses:

    print(mes)

    arquivos = os.listdir(path+'2021_'+mes)

    lista_arquivos = []

    for arquivo in arquivos:

        if arquivo.startswith('windBlowDust_PM10_2021-'+mes):

            nc = Dataset(path+'2021_'+mes+'/'+arquivo)

            print(nc.variables.keys())

            pm10 = nc.variables["PM10"][:]   # ajuste o nome se for diferente

            pm10_array = np.array(pm10)

            soma_total = np.nansum(pm10_array)

            lista_arquivos.append(soma_total)

    soma = np.nansum(lista_arquivos)

    print(soma)

    lista_meses.append(soma)

print(lista_meses)
print(lista_meses/np.max(lista_meses))

'''
