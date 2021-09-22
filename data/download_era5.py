#!/usr/bin/env python3

import cdsapi

request = {
    "product_type": "reanalysis",
    "variable": ["geopotential"],
    "grid": "1.0/1.0",
    "pressure_level": ["500"],
    "year": ["2016"],
    "month": ["03"],
    "day": ["13"], 
    "time": [ "00:00" ],
    "area": "90/-180/0/180",
    "format": "netcdf"
}

c = cdsapi.Client()
c.retrieve("reanalysis-era5-pressure-levels", request, "era5-2016-03-13.nc")

