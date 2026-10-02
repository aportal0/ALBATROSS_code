#!/usr/bin/env python3
"""
Dry-day frequency over Madagascar: ECMWF SEAS5 (24h tp) vs MSWEP, per month, per grid point.

Parallel version for ECWMF Atos. Run with, e.g.:

    srun --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G --time=01:00:00 \
         python drydays_parallel.py
"""

from pathlib import Path
import os
import numpy as np
import xarray as xr
from dask.distributed import Client, LocalCluster
import tempfile

def define_thr_label(fc_thr_m, ob_thr_mm):
    """ Define label for each combination of forecast and observed thresholds."""
    if fc_thr_m  == 0.0001 and ob_thr_mm == 0.1:
        return "thr0"
    elif fc_thr_m == 0.0002 and ob_thr_mm == 0.1:
        return "thr1"
    elif fc_thr_m == 0.0005 and ob_thr_mm == 0.1:
        return "thr2"
    elif fc_thr_m == 0.001 and ob_thr_mm == 0.1:
        return "thr3"
    elif fc_thr_m == 0.001 and ob_thr_mm == 1:
        return "thr4"
    else:
        print("No threshold label for the chosen combination of thresholds")
        return ""


# ----------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------
LOCAL_SCRATCH = os.environ.get("TMPDIR")
if not LOCAL_SCRATCH or not Path(LOCAL_SCRATCH).is_dir():
    LOCAL_SCRATCH = tempfile.gettempdir()
print(f"dask local_directory: {LOCAL_SCRATCH}", flush=True)

FC_DIR = Path("/ec/res4/scratch/ecme4047/C3S_seasonal/ecmwf51/24h/init_10/tp/regridded_ERA5-Land")
OB_DIR = Path("/ec/res4/scratch/ecme4047/MSWEP/MSWEP_V316_test/Past/Daily/regridded_ERA5-Land")
OUT_DIR = Path("/ec/res4/scratch/ecme4047/C3S_seasonal/ecmwf51/24h/init_10/tp/calibrated_MSWEP")

REGION       = "Madagascar"
YEARS        = range(1993, 2022 +1)
INIT_MONTH   = 10
N_MONTHS     = 3

N_MEMBERS_KEEP = 25          # members common to all years; extra post-2016 members dropped

FC_VAR       = "tp"
OB_VAR       = "precipitation"

FC_THRESH_M  = 0.001
OB_THRESH_MM = 0.1
THR_LABEL = define_thr_label(FC_THRESH_M, OB_THRESH_MM)

FC_TIME_DIM  = "valid_time"          # <- matches your fc files
OB_TIME_DIM  = "time"                # <- matches your MSWEP files
FC_CONCAT_DIM = "forecast_reference_time"   # <- as in your code

# parallelism settings
N_WORKERS    = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
TIME_CHUNK   = 40                    # timesteps per dask chunk
WORKER_MEM   = "3GB"
LOCAL_SCRATCH = os.environ.get("TMPDIR", "/local")

SPATIAL = ("lat", "latitude", "lon", "longitude")

OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = [
    (((INIT_MONTH - 1 + k) % 12) + 1, (INIT_MONTH - 1 + k) // 12)
    for k in range(N_MONTHS)
]


def dry_fraction(da: xr.DataArray, thresh: float) -> xr.DataArray:
    """Percentage of days below `thresh`, averaged over every non-spatial axis."""
    reduce_dims = [d for d in da.dims if d not in SPATIAL]
    valid = da.notnull()
    dry   = (da < thresh) & valid
    return (dry.sum(dim=reduce_dims) / valid.sum(dim=reduce_dims) * 100.0)


def keep_members(da, n=N_MEMBERS_KEEP, dim="number"):
    """Restrict the ensemble to the first n members, so years with extra members
    (after the 2016 init) do not enter the statistics with a different weight."""
    if dim in da.dims and da.sizes[dim] > n:
        da = da.isel({dim: slice(0, n)})
    return da


# start a distributed cluster; threads_per_worker=1 avoids GIL contention on netCDF reads
def main():
    cluster = LocalCluster(
        n_workers=N_WORKERS,
        threads_per_worker=1,
        memory_limit=WORKER_MEM,
        local_directory=LOCAL_SCRATCH,
        dashboard_address=None,
    )
    client = Client(cluster)
    print(f"dask: {client}", flush=True)
    try:
        # ----------------------------------------------------------------------
        # 1. forecast — one lazy dataset per calendar month
        #
        #    open_dataset per file is cheap; we keep it lazy so each file is only
        #    read once, when its chunk is actually needed by the final computation.
        # ----------------------------------------------------------------------
        fc_files = []
        for y in YEARS:
            f = FC_DIR / f"tp_24h_ecmwf51_init{y}{INIT_MONTH:02d}_{REGION}_res_ERA5-Land_patch.nc"
            if f.exists():
                fc_files.append(f)
            else:
                print(f"missing forecast: {f}")
        
        fc_monthly = {}
        for month, _ in TARGET:
            parts = []
            for f in fc_files:
                ds = xr.open_dataset(f, chunks={FC_TIME_DIM: TIME_CHUNK})   # lazy, chunked
                sub = ds[FC_VAR].sel({FC_TIME_DIM: ds[f"{FC_TIME_DIM}.month"] == month},
                                     drop=True)                            # empty months drop here
                sub = keep_members(sub)                                    # remove extra members after 2016 init
                parts.append(sub)                                     # NOT .load()
                parts[-1].encoding.pop("source", None)                # let dask own the read
            if not parts:
                print(f"forecast {month:02d}: no data")
                continue
            fc_monthly[month] = xr.concat(parts, dim=FC_CONCAT_DIM)
            print(f"forecast {month:02d}: {fc_monthly[month].sizes}")
        
        # ----------------------------------------------------------------------
        # 2. MSWEP — one lazy dataset per month
        # ----------------------------------------------------------------------
        ob_monthly = {}
        for month, yoff in TARGET:
            parts = []
            for y in YEARS:
                f = OB_DIR / f"precip_daily_{y + yoff}{month:02d}_{REGION}_res_ERA5-Land.nc"
                if not f.exists():
                    print(f"missing MSWEP: {f}")
                    continue
                ds = xr.open_dataset(f, chunks={OB_TIME_DIM: TIME_CHUNK})
                parts.append(ds[OB_VAR])
            if not parts:
                print(f"no MSWEP data for month {month:02d}")
                continue
            ob_monthly[month] = xr.concat(parts, dim=OB_TIME_DIM)
            print(f"MSWEP {month:02d}: {ob_monthly[month].sizes}")
        
        # ----------------------------------------------------------------------
        # 3. build every month's graph, then compute once and write
        # ----------------------------------------------------------------------
        for month, _ in TARGET:
            fields = {}
            if month in fc_monthly:
                fields["ecmwf51"] = dry_fraction(fc_monthly[month], FC_THRESH_M)
            if month in ob_monthly:
                fields["mswep"] = dry_fraction(ob_monthly[month], OB_THRESH_MM)
            if not fields:
                continue
        
            out = xr.Dataset(
                {name: da.rename(f"drydays_pct_{name}").astype("float32")
                 for name, da in fields.items()}
            )
            for name in out.data_vars:
                out[name].attrs.update(
                    units="%",
                    long_name=f"percentage of dry days ({name})",
                    threshold=str(FC_THRESH_M if name == "ecmwf51" else OB_THRESH_MM)
                )
            if "ecmwf51" in out and "mswep" in out:
                land_mask = out["ecmwf51"].notnull()
                out["mswep"] = out["mswep"].where(land_mask)
                out["bias"] = out["ecmwf51"] - out["mswep"]
                out["bias"].attrs.update(units="%",
                                         long_name="dry-day bias, forecast minus MSWEP")
        
        
            path = OUT_DIR / f"drydays_pct_mon{month:02d}_init{INIT_MONTH:02d}_{YEARS[0]}-{YEARS[-1]}_{REGION}_{THR_LABEL}.nc"
            # compute + write in one pass; dask reads each input file once
            out.to_netcdf(path, encoding={v: {"zlib": True, "complevel": 4}
                                          for v in out.data_vars})
            print(f"wrote {path}  |  " + "  ".join(
                f"{v}={float(out[v].mean().compute()):.1f}%" for v in out.data_vars
            ))
        
        client.close()
        cluster.close()
        print("done")
        
    finally:
        client.close()
        cluster.close()


if __name__ == "__main__":
    main()




