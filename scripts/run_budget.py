import sys

PROJECT_ROOT = "/home/mhpereir/eulerian_heat_budget"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from src import io, validate, config, grid
import numpy as np

path_data = "/home/mhpereir/downloads-mhpereir/ERA5_zg_PNW"

if __name__ == "__main__":
    # Example usage
    ds_T = io.load_era5_T(f"{path_data}/T.nc")
    ds_u = io.load_era5_u(f"{path_data}/ux.nc")
    ds_v = io.load_era5_u(f"{path_data}/uy.nc")
    ds_omega = io.load_era5_omega(f"{path_data}/uz.nc")
    ds_sfp = io.load_era5_sfp(f"{path_data}/sfp.nc") #surface pressure
    # Merge datasets on common coordinates and variables
    ds_merged = io.load_era5_merge_dataset(ds_T, ds_u, ds_v, ds_omega, ds_sfp)

    # Validate merged dataset against strict schema
    validate.validate_dataset(ds_merged)

    # Determine domain extent based on grid and config margin
    ds_domain = grid.determine_domain(ds_merged)
    
    # # Construct integand cell weights and masks
    ds_horizontal_cell_areas = grid.get_horizontal_cell_areas(ds_domain).astype("float64")
    ds_vertical_cell_areas = grid.get_vertical_cell_areas(ds_domain).astype("float64")

    ds_cell_volumes = grid.get_cell_volumes(ds_domain).astype("float64")

    print(ds_domain)

    print(ds_horizontal_cell_areas)

    print(ds_vertical_cell_areas['east'])

    print(ds_cell_volumes)

