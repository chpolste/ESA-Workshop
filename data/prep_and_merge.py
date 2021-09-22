#!/usr/bin/env python3

import numpy as np
import xarray as xr


# ERA5 reanalysis: Z500 field for forecast validation
era = xr.load_dataset("era5-2016-03-13.nc")
# IFS ENS perturbed forecasts initialized 2016-03-07 every 24 hours out to +144h
ens = xr.load_dataset("ens-2016-03-07.nc")

# Extract European region for validation
eur = dict(longitude=slice(-12.5, 42.5), latitude=slice(75, 35), time="2016-03-13T00")
era_z500_eur = era.sel(**eur, drop=True).z / 9.81
ens_z500_eur = ens.sel(**eur, drop=True).gh
del_z500_eur = ens_z500_eur - era_z500_eur

# Root mean square error in European region as forecast error metric
lat_weights = np.cos(np.deg2rad(era.latitude))
rmse_z500_eur = np.sqrt((del_z500_eur**2).weighted(lat_weights).mean(dim=("longitude", "latitude")))

# DataArray with RMSE
rmse_data = rmse_z500_eur.astype(np.float32)
rmse_coords = [ens.number.values]
rmse_dims = ["number"]
rmse_xr = xr.DataArray(data=rmse_data, coords=rmse_coords, dims=rmse_dims, name="rmse", attrs={
    "variable": "500 hPa Geopotential Height RMSE",
    "region": "Europe (35°N-75°N, 12.5°W-42.5°E)",
    "lead": "+144h",
    "units": "gpm"
})

# DataArray with Z500 forecast data
z500_xr = ens.gh.rename("z500").sel(longitude=slice(-180, 60), latitude=slice(90, 10))

# # Merge into Dataset, set metadata and write to disk
out = xr.merge([rmse_xr, z500_xr], combine_attrs="drop")
out.attrs["model"] = "IFS ENS"
out.attrs["init"] = "2016-03-07 00Z"
out.attrs["reanalysis"] = "ERA5"
out.attrs["model_data_info"] = "Model data obtained from TIGGE (https://apps.ecmwf.int/datasets/licences/tigge/) © 2021 European Centre for Medium-Range Weather Forecasts (ECMWF)"
out.attrs["reanalysis_data_info"] = "Reanalysis data obtained from Copernicus Climate Data Store (https://dx.doi.org/10.24381/cds.bd0915c6)"
out.to_netcdf("data-2016-03-07.nc")

