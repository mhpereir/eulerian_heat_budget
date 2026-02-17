import sys

PROJECT_ROOT = "/home/mhpereir/eulerian_heat_budget"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from src import io, validate, config, grid

path_data = "/home/mhpereir/downloads-mhpereir/ERA5_zg_PNW"

if __name__ == "__main__":
    # Example usage
    ds_T = io.load_era5_T(f"{path_data}/T.nc")
    ds_u = io.load_era5_u(f"{path_data}/ux.nc")
    ds_v = io.load_era5_u(f"{path_data}/uy.nc")
    ds_omega = io.load_era5_omega(f"{path_data}/uz.nc")
    ds_sfp = io.load_era5_sfp(f"{path_data}/sfp.nc") #surface pressure


    ds_merged = io.load_era5_merge_dataset(ds_T, ds_u, ds_v, ds_omega, ds_sfp)

    validate.validate_dataset(ds_merged)

    print(ds_merged)
