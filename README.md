# Dust in the Wind

A set of Python scripts to estimate particulate matter emissions from wind-blown dust (soil resuspension) for CMAQ modeling domains. The module processes WRF/MCIP meteorological data, soil properties (texture and density), and land use/land cover data (MapBiomas) to generate CMAQ-ready emission inventories, including chemical speciation of the particles.

## Scientific reference

The emission calculation is based on the formulation described in:

- LeGrand, S. L., Polashenski, C., Letcher, T. W., Creighton, G. A., Peckham, S. E., & Cetola, J. D. (2019). *The AFWA dust emission scheme for the GOCART aerosol model in WRF-Chem v3.8.1*. Geoscientific Model Development. [https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2016MS000823](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2016MS000823)

Additional references used for the threshold friction velocity and particle-size distribution routines:

- Kok, J. F. et al. (2010). *An improved dust emission model – Part 1: Model description and comparison against measurements*. [https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2010JD014649](https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2010JD014649)
- Vasques, G.M., Coelho, M.R., Dart, R.O., Cintra, L.C., Baca, J.F.M. (2021). *Soil Clay, Silt and Sand Content Maps for Brazil at 0-5, 5-15, 15-30, 30-60, 60-100 and 100-200 cm Depth Intervals with 90 m Spatial Resolution*. Embrapa Solos, Rio de Janeiro, Brazil.

## Script structure

| Script | Function |
|---|---|
| `shRunnerDustIntheWind.py` | Main script, executable from the command line. Takes the file paths and simulation parameters and orchestrates the entire pipeline. |
| `gridDetails.py` | Defines the modeling grid from the MCIP files (GRIDDOT2D/METCRO3D) and matches it with the WRF coordinates. |
| `regridMAPBIOMAS.py` | Regrids the land use/land cover data (MapBiomas) onto the modeling grid. |
| `soilPrep.py` | Prepares soil properties (clay, silt, sand content, bulk density) from Embrapa and IBGE data, and computes the particle-size distribution for each grid cell. |
| `metPrep.py` | Calculates surface roughness, friction velocity (ustar), and threshold friction velocities from WRF output. |
| `windBlowDustCalc.py` | Calculates the horizontal and vertical dust fluxes and the total dust emission for each grid cell. |
| `windBlowDustSpeciation.py` | Performs the chemical speciation of the estimated PM emissions. |
| `netCDFcreator.py` | Generates the output netCDF files in IO/API format, ready for use in CMAQ. |

## Requirements and installation

The setup workflow for the environment and input data is as follows:

1. **Install Python 3.12.9**
2. **Create a virtual environment** to isolate the project dependencies
3. **Install the packages** listed in `requirements.txt`
4. **Download the scripts** from this repository (GitHub)
5. **Download the input data**, required before the first run:
   - **WRF** and **MCIP** files (meteorology and grid definition)
   - **Embrapa** soil data (clay, silt, sand content, and bulk density)
   - **MapBiomas** land use/land cover file
   - **IBGE** soil species data

### Step-by-step

```bash
# 1. Install Python 3.12.9 (pyenv or the official installer is recommended)

# 2. Create the virtual environment
python3.12 -m venv venv
source venv/bin/activate

# 3. Install the dependencies
pip install -r requirements.txt

# 4. Clone the repository
git clone <repository-url>
cd DustIntheWind
```

After that, download and place the WRF/MCIP, Embrapa, MapBiomas, and IBGE data in the `inputs/` folder, following the paths expected by the `soilPrep.py` and `regridMAPBIOMAS.py` scripts.

## Running the model

The main script is run from the terminal, inside the virtual environment, with the following positional arguments:

```bash
python shRunnerDustIntheWind.py \
    /main/path \
    /outputs/path \
    path/with/mcip/archives \
    path/with/wrf/archives \
    domain_number \
    grid_name \
    year \
    True/False \
    yyyy-mm-dd
```

| Argument | Description |
|---|---|
| `windBlowDustFolder` | Path to the module's main (master) folder |
| `outfolder` | Path to the output results folder |
| `mcipPath` | Path to the MCIP files |
| `wrfoutFolder` | Path to the WRF output files |
| `domain` | WRF domain number |
| `GDNAM` | Grid name, as defined in MCIP |
| `YEAR` | Reference year of the simulation |
| `RESET_GRID` | `1` (True) to reprocess the intermediate grid/soil files, or `0` (False) to reuse the existing ones |
| `YYYYMMDD` | Reference date of the simulation |

### Internal processing workflow

1. `gridDetails.py` builds the modeling domain and matches the WRF dates with the MCIP dates.
2. `regridMAPBIOMAS.py` regrids the land use/land cover data onto the modeling grid.
3. `soilPrep.py` prepares clay, silt, sand content, and bulk density, along with the particle-size distribution for each grid cell.
4. For each particle diameter bin, `metPrep.py` computes the surface roughness and the observed and threshold friction velocities.
5. `windBlowDustCalc.py` estimates the horizontal and vertical dust fluxes and the total emission (g/s) per grid cell and per hour.
6. `windBlowDustSpeciation.py` chemically speciates the fine and coarse PM emissions.
7. `netCDFcreator.py` writes the raw and speciated results to netCDF files in IO/API format, ready for use in CMAQ.

## Outputs

Results are written to the output folder defined in `outfolder`, including:

- netCDF files with fine PM (`PMFINE`), coarse PM (`PMC`), and PM10 emissions per grid cell and per hour
- A netCDF file with the chemically speciated emissions
- Diagnostic figures (ustar, roughness, clay content, emission fluxes, etc.) for result verification
