#!/usr/bin/env python3
"""
Monthly aggregation of MSWEP daily files.

Daily files are named <year><doy>.nc, e.g. 2000001.nc ... 2000365.nc
(day-of-year, no zero padding on the year).

For each month:
  - use Past/Daily if the month is fully covered there
  - otherwise use NRT/Daily, and from that month onwards stay on NRT
"""

import glob
import os
import subprocess
import sys
from datetime import date, timedelta
import calendar

BASE = "/ec/res4/scratch/ecme4047/MSWEP/MSWEP_V316_test"

PAST_DAILY = os.path.join(BASE, "Past", "Daily")
PAST_MONTHLY = os.path.join(BASE, "Past", "Monthly")
NRT_DAILY = os.path.join(BASE, "NRT", "Daily")
NRT_MONTHLY = os.path.join(BASE, "NRT", "Monthly")

YEAR_START = 2025
YEAR_END = 2026


def doy(d):
    """Day of year, 1-based."""
    return (d - date(d.year, 1, 1)).days + 1


def month_days(year, month):
    first = date(year, month, 1)
    return first, date(year, month, calendar.monthrange(year, month)[1])


def daily_files(directory, year, month):
    """Daily files covering one month, in chronological order.

    Files are <year><doy>.nc; the day of year is NOT zero padded, so
    match the numeric DOY and range-check it rather than globbing widths.
    """
    first, last = month_days(year, month)
    lo, hi = doy(first), doy(last)

    found = {}
    for path in glob.glob(os.path.join(directory, f"{year}*.nc")):
        name = os.path.basename(path)
        stem = name[len(str(year)):-3]          # strip '<year>' and '.nc'
        if not stem.isdigit():
            continue
        n = int(stem)
        if lo <= n <= hi:
            found[n] = path

    return [found[n] for n in sorted(found)]


def month_is_complete(files, year, month):
    """True only if every day of the month is present."""
    first, last = month_days(year, month)
    expected = doy(last) - doy(first) + 1
    return len(files) == expected


def run_monsum(files, outfile):
    """cdo -O monsum -mergetime <files...> <outfile>"""
    cmd = ["cdo", "-O", "monsum", "-mergetime", *files, outfile]
    print(f"  cdo -O monsum -mergetime ({len(files)} files) -> {outfile}")
    subprocess.run(cmd, check=True)


def main():
    os.makedirs(PAST_MONTHLY, exist_ok=True)
    os.makedirs(NRT_MONTHLY, exist_ok=True)

    nrt_only = False   # flips to True the first month Past can't cover

    for year in range(YEAR_START, YEAR_END + 1):
        for month in range(1, 13):
            print(f"Processing {year}-{month:02d}")

            source_dir = NRT_DAILY if nrt_only else PAST_DAILY
            files = daily_files(source_dir, year, month)

            if not nrt_only and not month_is_complete(files, year, month):
                print(f"  Past/Daily incomplete or empty for {year}-{month:02d}"
                      f" -> switching to NRT from here on")
                nrt_only = True
                source_dir = NRT_DAILY
                files = daily_files(NRT_DAILY, year, month)

            if not files:
                print(f"  WARNING: no files for {year}-{month:02d} in {source_dir}, skipping")
                continue

            outdir = NRT_MONTHLY if source_dir == NRT_DAILY else PAST_MONTHLY
            outfile = os.path.join(outdir, f"{year}{month:02d}.nc")

            if os.path.exists(outfile):
                print(f"  exists, skipping: {outfile}")
                continue

            try:
                run_monsum(files, outfile)
                print(f"  wrote {outfile} from {len(files)} files")
            except subprocess.CalledProcessError as e:
                print(f"  ERROR: cdo failed for {year}-{month:02d}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()

